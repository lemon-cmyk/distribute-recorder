URL="https://download.docker.com/linux/ubuntu/gpg"
TRIES=3
TIMEOUT=10

url_reachable=0
for i in $(seq 1 $TRIES); do
  if wget --spider --timeout=$TIMEOUT --tries=1 "$URL"; then
    url_reachable=1
   break
  fi
  sleep 2
done

if [ $url_reachable -eq 1 ]; then
  echo "Found Docker download working. Do not need to show help dialog."
  exit 1
fi

echo "Found Docker download NOT working. Should show help dialog."
exit 0