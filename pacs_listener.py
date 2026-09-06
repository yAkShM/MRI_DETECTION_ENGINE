import os
import threading
from pynetdicom import AE, evt, StoragePresentationContexts

class PACSListener:
    def __init__(self, ae_title="BRAIN_AI_SCP", port=11112, on_series_complete_callback=None):
        self.ae_title = ae_title
        self.port = port
        self.on_series_complete = on_series_complete_callback
        
        self.ae = AE(ae_title=self.ae_title)
        # Support MR Image Storage and standard presentation contexts
        self.ae.supported_contexts = StoragePresentationContexts
        
        self.server = None
        self.is_running = False
        
        self.series_timers = {}
        self.series_paths = {}
        self.debounce_seconds = 3.0
        self.timer_lock = threading.Lock()
        
    def start(self):
        if self.is_running:
            return
            
        handlers = [(evt.EVT_C_STORE, self.handle_store)]
        
        # Start server in non-blocking mode
        self.server = self.ae.start_server(("", self.port), evt_handlers=handlers, block=False)
        self.is_running = True
        print(f"PACS Listener started on port {self.port} with AET {self.ae_title}")
        
    def stop(self):
        if self.server:
            self.server.shutdown()
            self.is_running = False
            print("PACS Listener stopped.")
            
    def handle_store(self, event):
        """Handler for EVT_C_STORE."""
        ds = event.dataset
        ds.file_meta = event.file_meta
        
        # DICOM tags
        study_uid = getattr(ds, "StudyInstanceUID", "UnknownStudy")
        series_uid = getattr(ds, "SeriesInstanceUID", "UnknownSeries")
        sop_uid = getattr(ds, "SOPInstanceUID", "UnknownSOP")
        
        # Save slice
        staging_dir = os.path.join("staging", study_uid, series_uid)
        os.makedirs(staging_dir, exist_ok=True)
        
        filepath = os.path.join(staging_dir, f"{sop_uid}.dcm")
        ds.save_as(filepath, write_like_original=False)
        
        # Debounce logic
        with self.timer_lock:
            if series_uid in self.series_timers:
                self.series_timers[series_uid].cancel()
                
            self.series_paths[series_uid] = (staging_dir, study_uid, series_uid)
            timer = threading.Timer(self.debounce_seconds, self._series_timeout, args=[series_uid])
            self.series_timers[series_uid] = timer
            timer.start()
            
        # Return 0x0000 (Success) status
        return 0x0000
        
    def _series_timeout(self, series_uid):
        """Called when no new slices arrive for `debounce_seconds`."""
        with self.timer_lock:
            if series_uid in self.series_timers:
                del self.series_timers[series_uid]
            staging_dir, study_uid, _ = self.series_paths.pop(series_uid, (None, None, None))
            
        if staging_dir and self.on_series_complete:
            # Trigger background processing
            threading.Thread(
                target=self.on_series_complete, 
                args=(staging_dir, study_uid, series_uid)
            ).start()

if __name__ == "__main__":
    # Test script entrypoint
    def dummy_callback(staging_dir, study, series):
        print(f"Series {series} completed in {staging_dir}")
        
    listener = PACSListener(on_series_complete_callback=dummy_callback)
    listener.start()
    
    import time
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        listener.stop()
