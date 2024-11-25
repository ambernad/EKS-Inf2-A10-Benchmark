# EKS-Inf2-A10-Benchmark
This is repository with scripts and EKS manifests to run Stable Diffusion 2 benchmark on EKS nodes with Amazon Inferentia2 and NVidia A10

# Implementation guide:
1. Use EC2 with DLAMI to compile model for Inf2. Save model artefacts locally on EC2. Very specific for each model. Better to look for pre-built compilation script
2. Use Deep Learning Containers. Modify DLC (add package, copy model from step1, define custom CMD) and create dockerfile. Example: Inf2/sd2-inf2-inference-dockerfile
3. Write python script where model  should be deployed as API. Use Flask or FastAPI. Example: s3://abernads-ml-artefacts/sd2_service.py . Use this script in CMD part in dockerfile
4. Build and run docker container locally on the same EC2 where model was compiled. Example: 
    <pre>docker build . -f sd2-inf2-inference-dockerfile -t 623387590579.dkr.ecr.us-east-2.amazonaws.com/sd-inf2:v0.3  
    docker run -it -p 5000:5000 —name pt17 —device=/dev/neuron0 623387590579.dkr.ecr.us-east-2.amazonaws.com/sd-inf2:v0.3  </pre>
5. To test model make calls like this: <pre>curl -X POST -H "Content-Type: application/json" -d '{"prompt":"pikachu in the hat"}' http://localhost:5000/inference </pre>  If local test is ok. Push container to ECR
6. Create EKS cluster. Useful link: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/containers/index.html?highlight=aws.amazon.com%2Fneuron%3A%201 . Important!! Use GPU AMI for nodes. Use eskctl to create (in console it’s currently not available). Install Neuron device plugin (need to expose neuron cores to pods). Example manifest to create nodegroup: Inf2/sd2_inf_nodegroup.yaml
7. Create deployment.yaml (use image from p.5, specify resources and custom entrypoint and args, if needed). Example: Inf2/sd2_deployment.yaml. Run kubectl apply -f sd2_deployment.yaml
8. Create service.yaml. Example: Inf2/sd2_service.py. Run kubectl apply -f sd2_service.yaml
9. PROFIT.
