# Download and untar extend model
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1XLBD8LSmXeMQ_-Khv6bSPcUhirNcz1SI' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1XLBD8LSmXeMQ_-Khv6bSPcUhirNcz1SI" -O extend-longformer-large.tar.gz && rm -rf /tmp/cookies.txt

tar xvzf extend-longformer-large.tar.gz && rm extend-longformer-large.tar.gz

pip install -r requirements
pip install --upgrade transformers
