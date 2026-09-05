import pickle
import json
from google.oauth2.credentials import Credentials

# 1. Load the valid JSON you just created
with open('coldcases_auth.json', 'r') as f:
    data = json.load(f)

# 2. Convert to a Credentials object
creds = Credentials.from_authorized_user_info(data)

# 3. Save it to the pickle file your script expects
with open('coldcases_pickle.pickle', 'wb') as f:
    pickle.dump(creds, f)

print("✅ 'coldcases_pickle.pickle' has been updated successfully!")