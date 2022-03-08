# ReID Deployment Documentation

## Architecture

The image below shows the architecture of the deployment:

![image info](./arch.png)

## Service Components

**Manager API Pod**: it exposes the API endpoints to interact with the other pods and with the service from the outside.

- Available for CPU.
- Folder with implementation of API calls `examples\multiedgetracking\reid_manager`.
- High-level API definition `lib/sedna/service/server/reid_manager.py`.
- Defined by the Dockerfile `multi-edge-tracking-external_api.Dockerfile`.
- It should be deployed in the cloud.

**FeReID Pod**: it takes care of feature extraction and ReID steps.

- Available for CPU and GPU.
- Folder with specific implementation `examples\multiedgetracking\fe_reid`.
- Base service in `lib/sedna/core/multi_edge_tracking/multi_edge_tracking.py`.
- Defined by the Dockerfile `multi-edge-tracking-feature-extraction-reid.Dockerfile` or `multi-edge-tracking-gpu-feature-extraction-reid.Dockerfile`.
- It loads the model defined by the CRD in the YAML file `yaml/models/model_feature_extraction.yaml` or `yaml/models/model_m3l.yaml`.
- It should be deployed in the cloud.

**Detection/Tracking Pods**: they take care of tracking objects coming from a video stream.

- Available for CPU and GPU.
- Folder with specific implementation `examples\multiedgetracking\detection`.
- AI model code in `examples/multiedgetracking/detection/estimator/bytetracker.py`.
- Base service in `lib/sedna/core/multi_edge_tracking/multi_edge_tracking.py`.
- Defined by the Dockerfile `multi-edge-tracking-detection.Dockerfile` or `multi-edge-tracking-gpu-detection.Dockerfile`.
- They load the model defined by the CRD in the YAML file `yaml/models/model_detection.yaml`.
- It should be deployed at the edge.

## Build

Go to the `sedna/examples` directory and run: `./build_image.sh -r <your-docker-private-repo> multiedgetracking` to build the Docker images. Remember to push the images to your own Docker repository!

NOTE: If you are planning on bulding the GPU version of the detector/tracker, this extra steps need to be performed **before** running the `build_image` command:

- `cd examples/`
- `git clone git://anongit.freedesktop.org/git/gstreamer/gst-plugins-bad`
- `cd gst-plugins-bad` 
- `git fetch origin 1.14`
- `git pull`
- `git checkout 1.14.5`

Then, at build time the Dockerfile will copy the clones folder into the container so that the build process can continue.

Run `make crds` in the `SEDNA_HOME` and then register the new CRDS for models and multiedgetrackingservice in the K8S cluster.

Build the GM `make gmimage` and restart the GM pod.

## Deployment

Before deploying, check the `yaml/multi-edge-tracking-service-manager-fused.yaml` and make sure to set properly the parameters such as nodeSelector, nodeName docker image repository, type of model, kafkaSupport etc..

You can use the `yaml/multi-edge-tracking-service-gpu-manager-fused.yaml` if you want to use the GPU-tuned Docker images for the detectors/trackers. 

For the pods running in the cloud, you need to use **nodeName** instead of nodeSelector. This is a limitation that will be addressed in a future version.

For the edge nodes, you can use nodeSelector. I recommend labeling them with the following command to have the detector/tracking pod running on the edge nodes automatically:

- `kubectl label nodes edge_node0 edge_node1 edge_node2 reid-role=detector`

**IMPORTANT**: Make sure to copy the AI model to the correct path on the hosts **BEFORE** starting the pods:

- The node running the detector should have the YoloX model in `"/data/ai_models/object_detection/pedestrians/yolox.pth"`
- The node running the FE_ReID, should have the required models: `"/data/ai_models/deep_eff_reid/weights/efficientnet_v2_model_25.pth"` or `"/data/ai_models/m3l/m3l.pth"` depending on which one it's loaded by the YAML configuration.

To deploy the application, run the following commands:

- Do only once: `kubectl create -f yaml/models/model_feature_extraction.yaml`
- Do only once: `kubectl create -f yaml/models/model_m3l.yaml`
- Do only once: `kubectl create -f yaml/models/model_detection.yaml`
- `kubectl create -f yaml/multi-edge-tracking-service-fused.yaml`

With the default deployment, you will have:

- 1 Manager API pod listening at port `9907`
- 1 Fe_ReID pod
- 3 Detect/Track pods

### Notes

- The non-fused deployment mode is not fully tested, do not use it.
- The Kafka mode works, but the results are not accesible by the manager-api. Do not use.

## Workflow

Execute the following API calls in sequence to test the application/service (replace the IP with the IP of the node where the manager-api-pod is running):

**STEP 1**: Add RTSP video sources for the tracking pods, one call per video address. This operation can be skipped if the RTSP stream addresses are hardcoded ahead of time in `examples/multiedgetracking/reid_manager/components/rtsp_dispatcher.py`.

- `curl -X POST http://7.182.9.110:9907/sedna/add_video_address --data '{"url":"rtsp://7.182.8.79/video/0", "camid":0, "receiver":"hostname"}'`
- `curl -X POST http://7.182.9.110:9907/sedna/add_video_address --data '{"url":"rtsp://7.182.8.79/video/1", "camid":1, "receiver":"hostname"}'`
- `curl -X POST http://7.182.9.110:9907/sedna/add_video_address --data '{"url":"rtsp://7.182.8.79/video/2", "camid":2, "receiver":"hostname"}'`

The field `receiver` must contain the `hostname` of the machine which will access the relative RTSP stream. For example, if your tracking pods are running on a machine with hostname `edge-1`, you will write `"receiver":"edge-1"`. The ENV variable defined in the YAML file while take care of injecting the correct hostname value into the tracking pods. In order for this to work, the `/etc/hosts` file has to be correctly configured on the host machine.

**STEP 2**: It is strongly recommended to upload first the images of the target you want to track/find before starting the RTSP video stream (especially if such streams are not genereated by a camera but rather from a video file). To do this, you have two options: the first is to send files directly using the command **A**, the second is to send base64 encoded images with the command **B**. It is important that the images use the same compression algorithm (e.g., jpg, png).

- **Option A**: `curl -X POST http://7.182.9.110:9907/sedna/set_app_details  -H 'Expect:' -F data='{"userID":"DEFAULT", "op_mode":"tracking", "threshold": 0.75, "queryImagesFromNative": []}' -F target=@vit_vid.png  target=@vit_vide2.png`
- **Option B (not fully tested)**: `curl -X POST http://7.182.9.110:9907/v1/person/tracking/live/identification  -H 'Expect:' --data '{"userID": "DEFAULT", "op_mode":"tracking", "threshold": 0.75, "queryImagesFromNative": [], "cameraIds": [], "isEnhanced": 0}'`

After this step, you can start the videos from your RTSP server. However, it's recommended to wait a few seconds for the pods to receive the new configuration (target images and op_mode).

**STEP 3**: Call `http://7.182.9.110:9907/sedna/get_reid_result?userid=DEFAULT` in a browser to fetch and see the ReID result or run the following command from a CLI:

- `curl -X GET http://7.182.9.110:9907/sedna/get_reid_result?userid=DEFAULT --output result.png`

There are other endpoints available for the manager-api pod, details are in `lib/sedna/service/server/reid_manager.py`.

### Notes

- For every received frame where we find the target, we stream the result to the RTMP server. The address is hardcoded in the `examples/multiedgetracking/reid_manager/worker.py` file. 
- If you want to flush the ReID result buffer, call `curl -X GET http://7.182.9.110:9907/sedna/clean_frame_buffer`
- If you want to check how many images are in the ReID result buffer, call `curl -X GET http://7.182.9.110:9907/sedna/get_reid_buffer_size`
- Once the RTSP video addresses are loaded by the pods, they cannot be changed. You need to restart the pods.
- It is possible to hardcode the stream addresses in the manager-api pod to simplify testing operations.