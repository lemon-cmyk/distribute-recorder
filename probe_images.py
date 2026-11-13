#!/usr/bin/env python3
"""Read-only simultaneous Image probe; run after sourcing the host ROS environment."""
import argparse
import json
import math
import os
from pathlib import Path
import time


def intervals(values):
    return [b-a for a, b in zip(values, values[1:])]


def summarize(record):
    arrivals, stamps = record['arrivals'], record['stamps']
    gaps = intervals(arrivals)
    stamp_gaps = intervals(stamps)
    return {
        'messages': len(arrivals),
        'arrival_hz': (len(arrivals)-1)/(arrivals[-1]-arrivals[0]) if len(arrivals)>1 and arrivals[-1]>arrivals[0] else None,
        'stamp_hz': (len(stamps)-1)/(stamps[-1]-stamps[0]) if len(stamps)>1 and stamps[-1]>stamps[0] else None,
        'max_arrival_gap_ms': max(gaps)*1000 if gaps else None,
        'arrival_gap_p95_ms': sorted(gaps)[max(0, math.ceil(.95*len(gaps))-1)]*1000 if gaps else None,
        'max_stamp_gap_ms': max(stamp_gaps)*1000 if stamp_gaps else None,
        'duplicate_adjacent_stamps': sum(g==0 for g in stamp_gaps),
        'backward_stamps': sum(g<0 for g in stamp_gaps),
        **record.get('metadata', {}),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--topics', required=True, help='one full sensor_msgs/msg/Image topic name per line')
    parser.add_argument('--seconds', type=float, default=60)
    parser.add_argument('--warmup', type=float, default=5)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    topics = list(dict.fromkeys(line.split('#',1)[0].strip() for line in Path(args.topics).read_text().splitlines()))
    topics = [t for t in topics if t]
    if not topics or any(not t.startswith('/') for t in topics):
        parser.error('supply at least one full topic name')
    if args.seconds<=0 or args.warmup<0:
        parser.error('seconds must be positive; warmup must be nonnegative')
    import rclpy
    from sensor_msgs.msg import Image
    from rclpy.qos import qos_profile_sensor_data
    rclpy.init()
    node = rclpy.create_node('migration_image_probe_'+str(os.getpid()))
    records = {t:{'arrivals':[], 'stamps':[]} for t in topics}
    start = time.monotonic()+args.warmup
    end = start+args.seconds

    def receive(topic, msg):
        now = time.monotonic()
        if now<start or now>end:
            return
        record = records[topic]
        record['arrivals'].append(now)
        record['stamps'].append(msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9)
        record['metadata'] = {'width':msg.width,'height':msg.height,'encoding':msg.encoding,'step':msg.step,'frame_bytes':len(msg.data),'frame_id':msg.header.frame_id}

    subs = [node.create_subscription(Image,t,lambda m,t=t:receive(t,m),qos_profile_sensor_data) for t in topics]
    try:
        while time.monotonic()<end:
            rclpy.spin_once(node,timeout_sec=min(.1,max(0,end-time.monotonic())))
        report = {'domain':os.getenv('ROS_DOMAIN_ID','0'),'rmw':os.getenv('RMW_IMPLEMENTATION'),'xml':os.getenv('FASTRTPS_DEFAULT_PROFILES_FILE'),'measurement_seconds':args.seconds,'warmup_seconds':args.warmup,'topics':{t:summarize(records[t]) for t in topics}}
        for t in topics:
            report['topics'][t]['publishers']=[{'node':i.node_name,'namespace':i.node_namespace,'gid':bytes(i.endpoint_gid).hex(),'reliability':str(i.qos_profile.reliability),'durability':str(i.qos_profile.durability)} for i in node.get_publishers_info_by_topic(t)]
        report['all_topics_received'] = all(r['messages']>=2 for r in report['topics'].values())
        report['note'] = 'Receiving messages is not a frequency/stability acceptance test; compare with each stream configuration and local baseline.'
        Path(args.output).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(report,ensure_ascii=False,indent=2))
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0 if report['all_topics_received'] else 2


if __name__=='__main__':
    raise SystemExit(main())
