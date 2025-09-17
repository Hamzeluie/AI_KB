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
        docker run -d --name kb-server-container --gpus all -v <path to qdrant_schedule_price_multilingual_opt>:/app/db -p 5003:8000 -e API_KEY="test" kb-server 


# Input/Output structure
input and output of the service in dictionary structure.

## input structure:


        {
        "kb_id": <knowledge base ID>,
        "owner_id": <owner ID>,
        "document": {<dictionary or text>}
        }


## output structure:

