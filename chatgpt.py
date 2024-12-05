# import openai_secret_manager
import openai


# Let's setup the API key
# assert "openai" in openai_secret_manager.get_services()
# secrets = openai_secret_manager.get_secrets("openai")
secrets = "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"
# the key is important to get the results
print(secrets)

# Let's install the required package
#!pip install openai
#!pip install catboost

# Now, let's generate some text

openai.api_key = secrets

prompt = (f"Make a list of astronomical observatories:")

completions = openai.Completion.create(
    engine="text-davinci-002",
    prompt=prompt,
    max_tokens=1024,
    n=1,
    stop=None,
    temperature=0.5,
)

message = completions.choices[0].text
print(message)