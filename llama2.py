import boto3
import json
import streamlit as st

prompt_data="""
Act as a Shakespeare and write a poem on Genertaive AI
"""

# Access AWS credentials from environment variables  [To RUN at LOCAL]
#aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
#aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
# Access AWS credentials from environment variables  [To RUN at streamlit]
aws_access_key_id = st.secrets["aws"]["AWS_ACCESS_KEY_ID"]
aws_secret_access_key = st.secrets["aws"]["AWS_SECRET_ACCESS_KEY"]
aws_region = st.secrets["aws"]["AWS_DEFAULT_REGION"]

bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-west-2",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)

payload={
    "prompt":"[INST]"+ prompt_data +"[/INST]",
    "max_gen_len":512,
    "temperature":0.5,
    "top_p":0.9
}
body=json.dumps(payload)
model_id="meta.llama3-70b-instruct-v1:0"
response=bedrock.invoke_model(
    body=body,
    modelId=model_id,
    accept="application/json",
    contentType="application/json"
)

response_body=json.loads(response.get("body").read())
repsonse_text=response_body['generation']
print(repsonse_text)