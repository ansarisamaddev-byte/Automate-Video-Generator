import pickle
import json
from google.oauth2.credentials import Credentials

# 1. Load the valid JSON you just created
with open('client_secret_warrior_ethos.json', 'r') as f:
    data = json.load(f)

# 2. Convert to a Credentials object
creds = Credentials.from_authorized_user_info(data)

# 3. Save it to the pickle file your script expects
with open('token.pickle', 'wb') as f:
    pickle.dump(creds, f)

print("✅ 'token.pickle' has been updated successfully!")