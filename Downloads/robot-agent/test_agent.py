import json
import unittest
from unittest.mock import MagicMock, patch
import agent


class ClientTests(unittest.TestCase):
    def config(self):
        return dict(robot_id='robot-001', device_id='dev-123', device_name='一号机器人-Thor',
                    ssh_user='nvidia', ssh_port=22, ssh_password='local-secret',
                    server_url='http://10.1.47.61:8080', server_ip='10.1.47.61')

    def network(self):
        return dict(management_ip='10.2.3.4', interface='wlan0', ips=['10.2.3.4'])

    def test_password_not_in_payload(self):
        payload = agent.make_payload(self.config(), self.network())
        self.assertNotIn('local-secret', json.dumps(payload))
        self.assertNotIn('ssh_password', payload)
        self.assertEqual(payload['role'], '一号机器人-Thor')

    def test_request_has_no_authorization_or_password(self):
        response = MagicMock(status=200)
        response.read.return_value = b'{"ok":true}'
        opener = MagicMock()
        opener.open.return_value.__enter__.return_value = response
        with patch.object(agent, 'build_opener', return_value=opener):
            agent.send_report(self.config(), agent.make_payload(self.config(), self.network()))
        request = opener.open.call_args.args[0]
        self.assertIsNone(request.get_header('Authorization'))
        self.assertNotIn(b'local-secret', request.data)

    def test_route_source_selected(self):
        data = [[{'dev': 'wlan0', 'prefsrc': '10.2.3.4'}],
                [{'addr_info': [{'family': 'inet', 'scope': 'global', 'local': '10.2.3.9'},
                                {'family': 'inet', 'scope': 'global', 'local': '10.2.3.4'}]}]]
        with patch.object(agent, 'ip_json', side_effect=data):
            self.assertEqual(agent.network_info(self.config())['management_ip'], '10.2.3.4')

    def test_rename_keeps_identity(self):
        c = self.config()
        before = agent.make_payload(c, self.network())
        c['device_name'] = '二楼测试-Thor'
        after = agent.make_payload(c, self.network())
        self.assertEqual(before['device_id'], after['device_id'])
        self.assertEqual(after['role'], '二楼测试-Thor')


if __name__ == '__main__':
    unittest.main()
