import boto3
import json
import os
import streamlit as st

prompt_data = """
Act as a Shakespeare and write a poem on Generative AI
"""

# Access AWS credentials from environment variables  [To RUN at LOCAL]
#aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
#aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
# Access AWS credentials from environment variables  [To RUN at streamlit]
aws_access_key_id = st.secrets["aws"]["AWS_ACCESS_KEY_ID"]
aws_secret_access_key = st.secrets["aws"]["AWS_SECRET_ACCESS_KEY"]
#aws_region = st.secrets["aws"]["AWS_DEFAULT_REGION"] 
#try

# Initialize the Bedrock client
bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-west-2",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)

# Prepare the payload with validated parameters

payload = {
"prompt": prompt_data,
"max_tokens_to_sample": 512,
"temperature": 0.5,
"top_k": 250,
"top_p": 1,
"stop_sequences": ["\n\nHuman:"]
}

body = json.dumps(payload)

# Specify the model ID you want to use
model_id = "anthropic.claude-v2:1"

# Invoke the model
try:
    print ("11111")
    response = bedrock.invoke_model(
        body=body,
        modelId=model_id,
        accept="application/json",
        contentType="application/json",
    )

    # Parse the response correctly
    response_body = json.loads(response['body'].read().decode('utf-8'))
    response_text = response_body.get("completions", [{}])[0].get("data", {}).get("text", "")
    print(response_text)

except Exception as e:
    print(f"An error occurred: {str(e)}")
