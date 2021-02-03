
if [ ! -d "data/SIMEPU" ]
then
    echo "SIMEPU data not found at 'data' directory. Downloading..."
    curl -O -J https://nextcloud.maparla.duckdns.org/s/yjf6Tn6yN3FEP6T/download
    mkdir -p data
    tar -zxf simepu.tar.gz  -C data/
    rm simepu.tar.gz
    echo "Done!"
else
  echo "SIMEPU already downloaded!"
fi