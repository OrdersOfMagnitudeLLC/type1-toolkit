# Copyright (c) 2026 Orders of Magnitude LLC
# Licensed under the OOM Commercial License v1.0
# See LICENSE.md in the repository root or ofmagnitude.com

#!/usr/bin/env python3
"""
Fetch specific CIF files from GNoME ZIP archive
Uses streaming to avoid downloading entire ZIP
"""

import requests
import zipfile
import io
import os

ZIP_URL = "https://storage.googleapis.com/gdm_materials_discovery/gnome_data/by_id.zip"

TARGET_IDS = [
    "fa37e9edd4",  # SiHF3
    "141d1a7b92",  # PH(OF)2
    "aab9f03c85",  # SrMgMn7O16
    "a953f4ace2",  # SrMn6Al2O16
    "91a39d50c2",  # Sr4MnFe3O10
    "79a464894e",  # MgS7
    "0da4f8a990",  # SrP2F12
]

OUT_DIR = os.path.join(os.path.expanduser('~'), "NS/Bob/data/cifs")

class StreamingZipFile:
    """Wrapper to allow zipfile to work with streaming requests"""
    def __init__(self, url):
        self.url = url
        self.response = requests.get(url, stream=True)
        self.response.raise_for_status()
        self.position = 0
        self.buffer = b''
        
    def read(self, size=-1):
        """Read from streaming response"""
        while size < 0 or len(self.buffer) < size:
            chunk = self.response.raw.read(8192)
            if not chunk:
                break
            self.buffer += chunk
        
        if size < 0:
            result = self.buffer
            self.buffer = b''
        else:
            result = self.buffer[:size]
            self.buffer = self.buffer[size:]
        
        self.position += len(result)
        return result
    
    def seek(self, offset, whence=0):
        """Seek in stream (limited functionality)"""
        if whence == 0:  # absolute
            if offset < self.position:
                raise ValueError("Cannot seek backwards in stream")
            while self.position < offset:
                chunk = self.read(8192)
                if not chunk:
                    break
        return self.position
    
    def tell(self):
        """Return current position"""
        return self.position

def main():
    print("Fetching CIF files from GNoME ZIP archive...")
    print(f"Target IDs: {TARGET_IDS}")
    print(f"Output directory: {OUT_DIR}\n")
    
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # Try streaming approach first
    print("Attempting streaming extraction...")
    try:
        # Use requests stream with zipfile
        response = requests.get(ZIP_URL, stream=True)
        response.raise_for_status()
        
        # Create a file-like object from the stream
        stream = io.BytesIO()
        downloaded = 0
        chunk_size = 1024 * 1024  # 1MB chunks
        
        for chunk in response.iter_content(chunk_size=chunk_size):
            stream.write(chunk)
            downloaded += len(chunk)
            print(f"\rDownloaded: {downloaded / 1e9:.2f} GB", end='', flush=True)
            
            # Try to parse as ZIP periodically
            stream.seek(0)
            try:
                with zipfile.ZipFile(stream, 'r') as zf:
                    namelist = zf.namelist()
                    found_count = 0
                    
                    for target_id in TARGET_IDS:
                        # Look for files containing our target ID
                        matching_files = [n for n in namelist if target_id in n]
                        
                        for match in matching_files:
                            if match.endswith('.cif') or match.endswith('.CIF'):
                                # Extract and save
                                data = zf.read(match)
                                output_path = os.path.join(OUT_DIR, f"{target_id}.cif")
                                with open(output_path, 'wb') as f:
                                    f.write(data)
                                print(f"\n✓ Extracted {target_id} from {match} ({len(data)} bytes)")
                                found_count += 1
                    
                    if found_count == len(TARGET_IDS):
                        print(f"\nAll {found_count} files found!")
                        return
            except:
                # ZIP not complete yet, continue downloading
                stream.seek(len(stream.getbuffer()))
                continue
        
        print("\nStreaming approach completed")
        
    except Exception as e:
        print(f"\nStreaming approach failed: {e}")
        print("Falling back to full download...")
        
        # Fallback: download full ZIP
        response = requests.get(ZIP_URL)
        response.raise_for_status()
        
        with zipfile.ZipFile(io.BytesIO(response.content), 'r') as zf:
            namelist = zf.namelist()
            
            for target_id in TARGET_IDS:
                matching_files = [n for n in namelist if target_id in n]
                
                for match in matching_files:
                    if match.endswith('.cif') or match.endswith('.CIF'):
                        data = zf.read(match)
                        output_path = os.path.join(OUT_DIR, f"{target_id}.cif")
                        with open(output_path, 'wb') as f:
                            f.write(data)
                        print(f"✓ Extracted {target_id} from {match} ({len(data)} bytes)")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
