
if [ ! -d "data/SIMEPU" ]
then
    echo "SIMEPU data not found at 'data' directory. Downloading..."
    curl -O -J https://nextcloud.maparla.duckdns.org/s/GLMQYE6ckcbYDJ2/download
    mkdir -p data
    tar -zxf simepu_ene29_2021_postAllMulti.tar.gz  -C data/
    rm simepu_ene29_2021_postAllMulti.tar.gz
    echo "Done!"
else
  echo "SIMEPU already downloaded!"
fi