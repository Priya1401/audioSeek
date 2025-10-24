### To run

```
docker build -t "audiobook-service" .
```

Once it builds, run this command

```
docker run -d \
  -p 8000:8000 \
  -p 8001:8001 \
  -e OPENAI_API_KEY="your-openai-api-key-here" \
  --name audiobook \
  audiobook-service
```


##### Try these commands

Find if the container is running

```
docker ps
```


Check logs

```
docker logs audiobook
```

If no errors, check some endpoints. You can find them in services/text_processing/controllers.py

