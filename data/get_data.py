from io import BytesIO # So we can treat bytes objects as files
import requests, tarfile, os
from bs4 import BeautifulSoup

url = "https://spamassassin.apache.org/old/publiccorpus/"
r = requests.get(url)
soup = BeautifulSoup(r.content)
for elt in soup.find_all('a'):
    fname = elt.get_text()
    if fname.startswith('200'):
        base = fname[:fname.find('.')] # get rid of .tar.bz
        # start untested changes
        dname = base[:4]
        if not os.path.exists(dname):
            os.mkdir(base)
        # end untested changes
        r = requests.get(url + fname)
        fp = BytesIO(r.content)
        with tarfile.open(fileobj=fp, mode='r:bz2') as tf:
            tf.extractall(dname)



