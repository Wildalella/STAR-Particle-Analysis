"""
ML Detection using Roboflow Inference SDK
Handles YOLOv8 model inference via Roboflow serverless API
"""

import json


def run_ml_detection(image_path, api_key="rhzcr5MSI8GCPCHm1OYv", 
                    workspace_name="payton-q3ubd", 
                    workflow_id="detect-count-and-visualize"):
    """
    Run Roboflow ML detection on image.
    
    Args:
        image_path: Path to image file
        api_key: Roboflow API key
        workspace_name: Roboflow workspace name
        workflow_id: Roboflow workflow ID
        
    Returns:
        list: Predictions, each with keys: x, y, width, height, confidence, class
              Returns empty list if detection fails
    """
    try:
        from inference_sdk import InferenceHTTPClient
    except ImportError:
        print("⚠ Warning: inference_sdk not installed")
        print("  Run: pip install inference-sdk")
        return []
    
    try:
        client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=api_key
        )
        
        # Run workflow
        result = client.run_workflow(
            workspace_name=workspace_name,
            workflow_id=workflow_id,
            images={"image": image_path},
            use_cache=True
        )
        
        # Extract predictions
        predictions_data = result[0].get("predictions", [])
        
        # Handle different response formats
        if isinstance(predictions_data, str):
            predictions_data = json.loads(predictions_data)
        
        if isinstance(predictions_data, dict) and "predictions" in predictions_data:
            predictions = predictions_data["predictions"]
        elif isinstance(predictions_data, list):
            predictions = predictions_data
        else:
            predictions = []
        
        return predictions
    
    except Exception as e:
        print(f"⚠ ML Detection failed: {e}")
        return []


def convert_predictions_to_centroids(predictions):
    """
    Convert Roboflow predictions to centroid coordinates.
    
    Args:
        predictions: List of prediction dicts from Roboflow
        
    Returns:
        list: List of (x, y) centroid tuples
    """
    centroids = []
    
    for pred in predictions:
        try:
            x = int(pred.get("x", 0))
            y = int(pred.get("y", 0))
            centroids.append((x, y))
        except (KeyError, TypeError, ValueError):
            continue
    
    return centroids
