import sys
import argparse
import httplib2

from oauth2client.client import flow_from_clientsecrets
from oauth2client.tools import argparser, run_flow
from oauth2client.file import Storage
from apiclient.discovery import build


CLIENT_SECRETS_FILE = "client_secrets.json"
YOUTUBE_READ_WRITE_SSL_SCOPE = "https://www.googleapis.com/auth/youtube.readonly"
API_SERVICE_NAME = "youtube"
API_VERSION = "v3"

def get_authenticated_service(args):
    flow = flow_from_clientsecrets(CLIENT_SECRETS_FILE, scope=YOUTUBE_READ_WRITE_SSL_SCOPE)
    storage = Storage("%s-oauth2.json" % sys.argv[0])
    credentials = storage.get()

    if credentials is None or credentials.invalid:
        credentials = run_flow(flow, storage, args)

  # Trusted testers can download this discovery document from the developers page
  # and it should be in the same directory with the code.
    return build(API_SERVICE_NAME, API_VERSION,
        http=credentials.authorize(httplib2.Http()))

args = argparser.parse_args()
service = get_authenticated_service(args)

def remove_empty_kwargs(**kwargs):
    good_kwargs = {}
    if kwargs is not None:
        for key, value in kwargs.iteritems():
          if value:
            good_kwargs[key] = value
    return good_kwargs


def search_list_by_keyword(service, **kwargs):
    kwargs = remove_empty_kwargs(**kwargs) # See full sample for function
    results = service.search().list(**kwargs).execute()
    return results


if __name__ == "__main__":
    # all_links = []
    for i in range(1,16):
        outfile = "res" + str(i) + ".txt"
        with open(outfile,"w") as output:
            infile = "play" + str(i) + ".list"
            output.write("%s\n" % infile)
            with open(infile) as f:
                for j in f:
                    output.write("%s\n" % j.strip())
                    res = search_list_by_keyword(service,
                        part='snippet',
                        maxResults=1,
                        q=j.strip(),
                        type='video')
                    # print(res)
                    vid = res["items"][0]["id"]["videoId"]
                    # all_links.append("http://www.youtube.com/watch?v=" + str(vid))
                    output.write("%s\n" % ("http://www.youtube.com/watch?v=" + str(vid)))