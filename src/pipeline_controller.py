import concurrent.futures
from typing import List
from .rawgl_wrapper import RawGLWrapper

class PipelineController:
    """
    Manages a multithreaded pipeline for executing RawGL processing jobs.
    """

    def __init__(self, max_workers: int = 4):
        """
        Initializes the pipeline controller.

        Args:
            max_workers (int): The maximum number of concurrent threads to run.
        """
        self.max_workers = max_workers
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        self.futures = []
        print(f"PipelineController initialized with {max_workers} worker threads.")

    def submit_job(self, rawgl_job: RawGLWrapper):
        """
        Submits a configured RawGLWrapper instance as a job to the thread pool.

        Args:
            rawgl_job (RawGLWrapper): A configured RawGLWrapper instance ready to be run.

        Returns:
            concurrent.futures.Future: A future object representing the execution of the job.
        """
        if not isinstance(rawgl_job, RawGLWrapper):
            raise TypeError("Job must be an instance of RawGLWrapper.")

        print(f"Submitting job: {rawgl_job.build_command()}")
        future = self.executor.submit(rawgl_job.run)
        self.futures.append(future)
        return future

    def wait_for_completion(self):
        """Blocks until all submitted jobs have completed."""
        print("Waiting for all submitted jobs to complete...")
        concurrent.futures.wait(self.futures)
        print("All jobs completed.")

        # Optional: retrieve and print results
        for future in self.futures:
            success, output = future.result()
            print(f"Job Result (Success: {success}):\n---\n{output}\n---")

    def shutdown(self, wait: bool = True):
        """
        Shuts down the thread pool executor.

        Args:
            wait (bool): If True, waits for all pending futures to complete before shutting down.
        """
        print("Shutting down the pipeline controller...")
        self.executor.shutdown(wait=wait)
        print("Shutdown complete.")


if __name__ == '__main__':
    # This example demonstrates how to use the PipelineController.
    # It creates a couple of mock jobs and runs them.

    print("--- Running PipelineController Example ---")

    # This is a placeholder path.
    RAWGL_PATH = "path/to/rawgl"

    # 1. Create a controller
    controller = PipelineController(max_workers=2)

    # 2. Create and configure two separate jobs
    job1 = RawGLWrapper(RAWGL_PATH)
    job1.add_pass(
        shader_path="shader1.frag",
        output_size=(1024, 1024),
        output_path="output1.png",
        inputs={'tex': 'input1.png'},
        is_greyscale_png=True
    )

    job2 = RawGLWrapper(RAWGL_PATH)
    job2.add_pass(
        shader_path="shader2.frag",
        output_size=(512, 512),
        output_path="output2.png",
        inputs={'tex': 'input2.png'}
    )

    # 3. Submit jobs to the controller
    future1 = controller.submit_job(job1)
    future2 = controller.submit_job(job2)

    # The jobs are now running in the background.
    # We can do other work here, or we can wait for them to finish.

    # 4. Wait for jobs to complete and then shut down
    controller.wait_for_completion()
    controller.shutdown()

    print("\n--- PipelineController Example Finished ---")
