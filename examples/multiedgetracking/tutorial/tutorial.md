# ReID Deployment Documentation

## Architecture

The image below shows the architecture of the deployment:

![image info](./arch.png)

## Service Components

**Manager API Pod**: it exposes the API endpoints to interact with the other pods and with the service from the outside.

- Folder with implementation of API calls `examples\multiedgetracking\reid_manager`
- High-level API definition `lib/sedna/service/server/reid_manager.py`
- Defined by the Dockerfile `multi-edge-tracking-external_api.Dockerfile`
- It should be deployed in the cloud

**FeReID Pod**: it takes care of feature extraction and ReID steps.

- Folder with specific implementation `examples\multiedgetracking\fe_reid`
- Base service in `lib/sedna/core/multi_edge_tracking/multi_edge_tracking.py`
- Defined by the Dockerfile `multi-edge-tracking-feature-extraction-reid.Dockerfile`
- It loads the model defined by the CRD in the YAML file `model_feature_extraction.yaml`
- It should be deployed in the cloud

**Detection/Tracking Pods**: they take care of tracking objects coming from a video stream.

- Folder with specific implementation `examples\multiedgetracking\detection`
- AI model code in `examples/multiedgetracking/detection/estimator/bytetracker.py`
- Base service in `lib/sedna/core/multi_edge_tracking/multi_edge_tracking.py`
- Defined by the Dockerfile `multi-edge-tracking-detection.Dockerfile`
- They load the model defined by the CRD in the YAML file `model_detection.yaml`
- It should be deployed at the edge

## Build

Go to the `sedna/examples` directory and run: `./build_image.sh -r <your-docker-private-repo> multiedgetracking` to build the Docker images. Remember to push the images to your own Docker repository!

Run `make crds` in the `SEDNA_HOME` and then register the new CRDS for models and multiedgetrackingservice in the K8S cluster.

Build the GM `make gmimage` and restart the GM pod.

## Deployment

Before deploying, check the `multi-edge-tracking-service-fused.yaml` and make sure to set properly the parameters such as nodeSelector, nodeName docker image repository, kafkaSupport etc..

For the pods running in the cloud, you need to use **nodeName** instead of nodeSelector. This is a limitation that will be addressed in a future version.

For the edge nodes, you can use nodeSelector. I recommend labeling them with the following command to have the detector/tracking pod running on the edge nodes automatically:

- `kubectl label nodes edge_node0 edge_node1 edge_node2 reid-role=detector`

**IMPORTANT**: Make sure to copy the AI model to the correct path on the hosts **BEFORE** starting the pods:

- The node running the detector should have the YoloX model in `"/data/ai_models/object_detection/pedestrians/yolox.pth"`
- The node running the FE_ReID, should have the model in `"/data/ai_models/deep_eff_reid/weights/efficientnet_v2_model_25.pth"`

To deploy the application, run the following commands:

- Do only once: `kubectl create -f model_feature_extraction.yaml`
- Do only once: `kubectl create -f model_detection.yaml`
- `kubectl create -f multi-edge-tracking-service-fused.yaml`

With the default deployment, you will have:

- 1 Manager API pod listening at port `9907`
- 1 Fe_ReID pod
- 3 Detect/Track pods

### Notes

- The non-fused deployment mode is not fully tested, do not use it.
- The Kafka mode works, but the results are not accesible by the manager-api. Do not use.

## Workflow

Execute the following API calls in sequence to test the application/service (replace the IP with the IP of the node where the manager-api-pod is running):

**STEP 1**: Add RTSP video sources for the tracking pods, 1 call per video address. This operation can be skipped if the RTSP stream addresses are hardcoded ahead of time in `examples/multiedgetracking/reid_manager/components/rtsp_dispatcher.py`.

- `curl -X POST http://7.182.9.110:9907/sedna/add_video_address --data '{"camera_address":"rtsp://localhost:8080/video/0", "camera_id":0}'`
- `curl -X POST http://7.182.9.110:9907/sedna/add_video_address --data '{"camera_address":"rtsp://localhost:8080/video/1", "camera_id":1}'`
- `curl -X POST http://7.182.9.110:9907/sedna/add_video_address --data '{"camera_address":"rtsp://localhost:8080/video/2", "camera_id":2}'`

**STEP 2**: Upload the images of the target you want to track/find. You have two options: the first is to send files directly using call **A**, the second is to send base64 encoded images with call **B**. It is important that the images use the same compression algorithms (e.g., jpg, png).

- **Option A**: `curl -X POST http://7.182.9.110:9907/sedna/set_app_details  -H 'Expect:' -F data='{"userID":"123", "op_mode":"tracking", "queryImagesFromNative": []}' -F target=@vit_vid.png  target=@vit_vide2.png`
- **Option B (not fully tested)**: `curl -X POST http://7.182.9.110:9907/v1/person/tracking/live/identification  -H 'Expect:' --data '{"userID": "123", "op_mode":"tracking", "queryImagesFromNative": [], "cameraIds": [], "isEnhanced": 0}'`

After this step, you can start the videos from your RTSP server. However, it's recommended to wait a few seconds for the pods to receive the new configuration (target images and op_mode).

**STEP 3**: Call this endpoint in a browser to fetch and see the ReID result from the FIFO queue:

- `curl -X GET http://7.182.9.110:9907/sedna/get_reid_result`

There are other endpoints available for the manager-api pod, details are in `lib/sedna/service/server/reid_manager.py`.

### Notes

- For every received frame where we find the target, we stream the result to the RTMP server. The address is hardcoded in the `examples/multiedgetracking/reid_manager/worker.py` file. 
- If you want to flush the frame buffer, call `curl -X GET http://7.182.9.110:9907/sedna/clean_frame_buffer`
- If you want to check how many images are in the buffer, call `curl -X GET http://7.182.9.110:9907/sedna/get_reid_buffer_size`
- Once the RTSP video addresses are loaded by the pods, they cannot be changed. You need to restart the pods.
- It is possible to hardcode the stream addresses in the manager-api pod to simplify testing operations.