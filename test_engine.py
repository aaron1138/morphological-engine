from dask_engine import DaskEngine

def test_dask_engine():
    print("Testing Dask Engine...")
    try:
        engine = DaskEngine(thread_count=2)
        engine.start_dask_client()
        dask_array = engine.load_images_to_dask_array("test_images")
        print(f"Dask array loaded successfully.")
        print(f"Shape: {dask_array.shape}")
        print(f"Chunk size: {dask_array.chunksize}")
        engine.stop_dask_client()
        print("Dask client stopped.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    test_dask_engine()
