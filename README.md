# AI_KB
this repository is knowledge base data base
## Repository files

**main.py**: script of main service.

**vector_db.py**: contain **MultiTenantVectorDB** class

**test_create**: create test bash script

**test_get**: get test bash script

**test_delete**: delete test bash script

**.env(optional)**: define environment variables in this file

## parameters

**HOST**: service host that you can assign in environment variable..

**PORT**: serice port that you can assign in environment variable.


# Install .venv
you can install virtual environment with **poetry**

        poetry install --no-root

# Run tests
test main function:

        . test_create.sh
        . test_get.sh
        . test_delete.sh


or you can run the main.py on host 0.0.0.0 and port 8000 with

        poetry run python main.py



# Docker
you can build docker image and run a container with the image.
Docker file Expose port is **8000**

with:


        docker build --no-cache -t kb-server .
        docker run -d --name kb-server-container --gpus all -v /home/ubuntu/borhan/whole_pipeline/vexu/AI_KB/db:/app/db -p 5003:8000 -e API_KEY="test" kb-server 


# Input/Output structure
there are four async function.

## **create_document**: 
post http request. endpoint structure "db/<owner_id>". create new knwledge base.

### input structure:

        
        {
                "kb_id": "string",
                "owner_id": "string",
                "document": 
                        {
                        "key1": {
                                "field1": "value1",
                                "field2": "value2"
                                },
                        "key2": {
                                "field1": "value1",
                                "field2": "value2"
                                }
                        }
        }

### output structure


        {
        "status": "success",
        "document": "Document created successfully",
        "kb_id": "string",
        "owner_id": "string"
        }

## **get_document**: 
get http request. endpoint structure "db/<owner_id>". apply a query with the **owner_id** to get the knowledge base.
### input structure

        {
        "query_text": "string",
        "kb_id": ["string"],
        "limit": integer
        }
### output structure


        {
                "status": "success",
                "owner_id": string,
                "documents": string,
                "count": integer,
        }


## **delete_document**: 
delete http request. endpoint structure "/db/<owner_id>/<kb_id>". delete a knowledge base.
### input structure

        "http://<HOST>:<PORT>/db/<OWNER_ID>/<KB_ID>"
### output structure

        {
                "status": "success", 
                "owner_id": string, 
                "message": "Document deleted"
        }

## **websocket_search**: 
websocket realtime search. endpoint structure "/ws/search/<owner_id>". 
### input structure

        {
                "query_text": string,
                "kb_id": string,
                "limit": integer,
        }
        
### output structure

        {
                "status": "success",
                "owner_id": string,
                "kb_id": string,
                "query": string,
                "results": 
                        {
                                "score": string,
                                "content": string,
                                "metadata": string,
                        }
                ,
                "count": integer,
        }

