from sedna.core.joint_inference import BigModelService
from interface import Estimator

def run():
    inference_instance = BigModelService(estimator=Estimator)
    inference_instance.start()

if __name__ == "__main__":
    run() 