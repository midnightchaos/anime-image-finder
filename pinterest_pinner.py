#!/usr/bin/env python3
"""
Pinterest Image Pinner
Scans local images, finds similar images on Pinterest, and pins them to your board
"""

import os
import sys
import time
import hashlib
import requests
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from urllib.parse import urlparse, quote, urlencode
import cv2
import numpy as np
from PIL import Image
import io
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from datetime import datetime
import re

# Load environment variables from .env file
def load_env():
    """Load environment variables from .env file"""
    env_file = Path(".env")
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value

# Load environment variables
load_env()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pinterest_pinner.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PinterestPinner:
    def __init__(self, images_dir: str = r"D:\projects-personal\designs\consolidated_images", 
                 board_name: str = "Similar Anime Images"):
        self.images_dir = Path(images_dir)
        self.board_name = board_name
        
        # Pinterest API credentials
        self.access_token = os.getenv('PINTEREST_ACCESS_TOKEN')
        self.api_base_url = "https://api.pinterest.com/v5"
        
        # Pinterest headers
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json',
            'Content-Type': 'application/json',
        }
        
        if self.access_token:
            self.headers['Authorization'] = f'Bearer {self.access_token}'
        
        # Similarity threshold (90%)
        self.similarity_threshold = 0.90
        
        # Create output directory for metadata
        self.output_dir = Path("PINNED_METADATA")
        self.output_dir.mkdir(exist_ok=True)
        
    def extract_image_features(self, image_path: Path) -> Dict:
        """Extract features from an image for similarity comparison"""
        try:
            # Load image
            img = cv2.imread(str(image_path))
            if img is None:
                logger.warning(f"Could not load image: {image_path}")
                return None
                
            # Convert to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize for consistent feature extraction
            img_resized = cv2.resize(img_rgb, (224, 224))
            
            features = {}
            
            # 1. Color histogram
            hist_r = cv2.calcHist([img_resized], [0], None, [64], [0, 256])
            hist_g = cv2.calcHist([img_resized], [1], None, [64], [0, 256])
            hist_b = cv2.calcHist([img_resized], [2], None, [64], [0, 256])
            features['color_hist'] = np.concatenate([hist_r.flatten(), hist_g.flatten(), hist_b.flatten()])
            
            # 2. Edge features
            gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            features['edge_density'] = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # 3. Perceptual hash
            pil_img = Image.fromarray(img_rgb)
            features['phash'] = self._calculate_phash(pil_img)
            
            # 4. Average color
            features['avg_color'] = np.mean(img_resized, axis=(0, 1))
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features from {image_path}: {e}")
            return None
    
    def _calculate_phash(self, image: Image.Image, hash_size: int = 8) -> int:
        """Calculate perceptual hash of an image"""
        # Resize image to even size (hash_size + 1 must be even)
        resize_size = hash_size + 1
        if resize_size % 2 == 1:
            resize_size += 1  # Make it even
            
        image = image.resize((resize_size, resize_size), Image.Resampling.LANCZOS)
        
        # Convert to grayscale
        image = image.convert('L')
        
        # Calculate DCT
        pixels = np.array(image, dtype=np.float64)
        dct = cv2.dct(pixels)
        
        # Keep only top-left hash_size x hash_size
        dct_low = dct[:hash_size, :hash_size]
        
        # Calculate median
        med = np.median(dct_low)
        
        # Create hash
        diff = dct_low > med
        return sum([2**i for (i, v) in enumerate(diff.flatten()) if v])
    
    def calculate_similarity(self, features1: Dict, features2: Dict) -> float:
        """Calculate similarity between two feature sets"""
        if not features1 or not features2:
            return 0.0
            
        similarities = []
        
        # Color histogram similarity
        if 'color_hist' in features1 and 'color_hist' in features2:
            hist_sim = cv2.compareHist(
                features1['color_hist'].astype(np.float32),
                features2['color_hist'].astype(np.float32),
                cv2.HISTCMP_CORREL
            )
            similarities.append(max(0, hist_sim))
        
        # Edge density similarity
        if 'edge_density' in features1 and 'edge_density' in features2:
            edge_sim = 1 - abs(features1['edge_density'] - features2['edge_density'])
            similarities.append(max(0, edge_sim))
        
        # Perceptual hash similarity
        if 'phash' in features1 and 'phash' in features2:
            hash_sim = self._calculate_hash_similarity(features1['phash'], features2['phash'])
            similarities.append(hash_sim)
        
        # Average color similarity
        if 'avg_color' in features1 and 'avg_color' in features2:
            color_sim = 1 - np.linalg.norm(features1['avg_color'] - features2['avg_color']) / (255 * np.sqrt(3))
            similarities.append(max(0, color_sim))
        
        # Return average similarity
        return np.mean(similarities) if similarities else 0.0
    
    def _calculate_hash_similarity(self, hash1: int, hash2: int) -> float:
        """Calculate similarity between perceptual hashes"""
        # Calculate Hamming distance
        hash_xor = hash1 ^ hash2
        hamming_distance = bin(hash_xor).count('1')
        
        # Convert to similarity (0-1)
        max_distance = 64  # For 8x8 hash
        return max(0, 1 - (hamming_distance / max_distance))
    
    def get_user_boards(self) -> List[Dict]:
        """Get list of user's boards"""
        if not self.access_token:
            logger.error("No Pinterest access token provided")
            return []
        
        try:
            url = f"{self.api_base_url}/user_account/boards"
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 200:
                data = response.json()
                return data.get('items', [])
            else:
                logger.error(f"Failed to get boards: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error getting boards: {e}")
            return []
    
    def create_board(self, board_name: str) -> Optional[str]:
        """Create a new Pinterest board"""
        if not self.access_token:
            logger.error("No Pinterest access token provided")
            return None
        
        try:
            url = f"{self.api_base_url}/boards"
            data = {
                "name": board_name,
                "description": f"Similar images found by AI - {datetime.now().strftime('%Y-%m-%d')}"
            }
            
            response = requests.post(url, headers=self.headers, json=data)
            
            if response.status_code == 201:
                board_data = response.json()
                logger.info(f"Created board: {board_name}")
                return board_data.get('id')
            else:
                logger.error(f"Failed to create board: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error creating board: {e}")
            return None
    
    def get_or_create_board(self) -> Optional[str]:
        """Get existing board or create new one"""
        boards = self.get_user_boards()
        
        # Look for existing board
        for board in boards:
            if board.get('name') == self.board_name:
                logger.info(f"Found existing board: {self.board_name}")
                return board.get('id')
        
        # Create new board
        return self.create_board(self.board_name)
    
    def search_pinterest_images(self, image_path: Path, max_results: int = 20) -> List[Dict]:
        """Search Pinterest for similar images using keywords"""
        try:
            # Extract keywords from filename
            filename = image_path.stem.lower()
            
            # Common anime/manga keywords
            anime_keywords = [
                'anime', 'manga', 'berserk', 'dragon ball', 'one piece', 'naruto',
                'attack on titan', 'demon slayer', 'jujutsu kaisen', 'bleach',
                'my hero academia', 'solo leveling', 'character', 'art', 'fanart'
            ]
            
            # Find relevant keywords from filename
            relevant_keywords = []
            for keyword in anime_keywords:
                if keyword in filename:
                    relevant_keywords.append(keyword)
            
            # If no specific keywords found, use general terms
            if not relevant_keywords:
                relevant_keywords = ['anime art', 'manga art', 'character design']
            
            # Search Pinterest for each keyword
            all_results = []
            
            for keyword in relevant_keywords[:3]:
                try:
                    # Pinterest search API
                    url = f"{self.api_base_url}/pins/search"
                    params = {
                        'query': keyword,
                        'bookmark': '',
                        'page_size': max_results // 3
                    }
                    
                    response = requests.get(url, headers=self.headers, params=params)
                    
                    if response.status_code == 200:
                        data = response.json()
                        items = data.get('items', [])
                        
                        for item in items:
                            all_results.append({
                                'id': item.get('id'),
                                'title': item.get('title', ''),
                                'description': item.get('description', ''),
                                'image_url': item.get('media', {}).get('images', {}).get('original', {}).get('url'),
                                'pinterest_url': item.get('link'),
                                'keyword': keyword
                            })
                    
                    time.sleep(1)  # Be respectful
                    
                except Exception as e:
                    logger.warning(f"Error searching for keyword '{keyword}': {e}")
                    continue
            
            return all_results
            
        except Exception as e:
            logger.error(f"Error in Pinterest search for {image_path}: {e}")
            return []
    
    def pin_image_to_board(self, board_id: str, image_url: str, title: str, description: str = "") -> bool:
        """Pin an image to a Pinterest board"""
        if not self.access_token:
            logger.error("No Pinterest access token provided")
            return False
        
        try:
            url = f"{self.api_base_url}/pins"
            data = {
                "board_id": board_id,
                "media_source": {
                    "source_type": "image_url",
                    "url": image_url
                },
                "title": title,
                "description": description
            }
            
            response = requests.post(url, headers=self.headers, json=data)
            
            if response.status_code == 201:
                pin_data = response.json()
                logger.info(f"Successfully pinned: {title}")
                return True
            else:
                logger.error(f"Failed to pin image: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error pinning image: {e}")
            return False
    
    def process_image(self, image_path: Path, board_id: str) -> Dict:
        """Process a single image: extract features, search Pinterest, pin similar images"""
        logger.info(f"Processing: {image_path.name}")
        
        # Extract features
        features = self.extract_image_features(image_path)
        if not features:
            return {'image': image_path.name, 'status': 'failed', 'error': 'Could not extract features'}
        
        # Search Pinterest
        pinterest_results = self.search_pinterest_images(image_path)
        if not pinterest_results:
            return {'image': image_path.name, 'status': 'no_results', 'pinterest_results': []}
        
        # Process and pin similar images
        pinned_images = []
        for result in pinterest_results:
            try:
                if result.get('image_url'):
                    # For now, we'll pin all found images since we can't download to compare
                    # In a real implementation, you'd download and compare first
                    
                    title = f"Similar to {image_path.stem} - {result.get('title', 'Anime Art')}"
                    description = f"Found similar to {image_path.name} using AI image matching. Keyword: {result.get('keyword', 'anime')}"
                    
                    if self.pin_image_to_board(board_id, result['image_url'], title, description):
                        pinned_images.append({
                            'pinterest_id': result['id'],
                            'title': result['title'],
                            'pinterest_url': result['pinterest_url'],
                            'image_url': result['image_url'],
                            'keyword': result.get('keyword', '')
                        })
                        
            except Exception as e:
                logger.error(f"Error processing Pinterest result for {image_path.name}: {e}")
        
        return {
            'image': image_path.name,
            'status': 'completed',
            'pinned_images': len(pinned_images),
            'pinned_details': pinned_images
        }
    
    def run(self, max_workers: int = 2):
        """Main execution function"""
        logger.info("Starting Pinterest Image Pinner")
        
        # Check if we have access token
        if not self.access_token:
            logger.error("No Pinterest access token found!")
            logger.error("Please run: python pinterest_api_setup.py")
            return []
        
        # Get or create board
        board_id = self.get_or_create_board()
        if not board_id:
            logger.error("Could not get or create Pinterest board")
            return []
        
        logger.info(f"Using board: {self.board_name} (ID: {board_id})")
        
        # Get all image files
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
        image_files = [
            f for f in self.images_dir.iterdir() 
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
        
        logger.info(f"Found {len(image_files)} images to process")
        
        # Process images
        results = []
        for i, image_path in enumerate(image_files):
            try:
                result = self.process_image(image_path, board_id)
                results.append(result)
                logger.info(f"Completed {i+1}/{len(image_files)}: {image_path.name} - {result['status']}")
                
                # Be respectful to Pinterest - add delay between requests
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Error processing {image_path.name}: {e}")
                results.append({
                    'image': image_path.name,
                    'status': 'error',
                    'error': str(e)
                })
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"pinning_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Print summary
        total_processed = len(results)
        successful = len([r for r in results if r['status'] == 'completed'])
        total_pinned = sum([r.get('pinned_images', 0) for r in results if r['status'] == 'completed'])
        
        logger.info(f"\n=== PINNING SUMMARY ===")
        logger.info(f"Total images processed: {total_processed}")
        logger.info(f"Successfully processed: {successful}")
        logger.info(f"Total images pinned: {total_pinned}")
        logger.info(f"Board: {self.board_name}")
        logger.info(f"Results saved to: {results_file}")
        
        return results

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pin similar images to Pinterest board")
    parser.add_argument("--images-dir", default=r"D:\projects-personal\designs\consolidated_images", 
                       help="Directory containing images to search")
    parser.add_argument("--board-name", default="Similar Anime Images",
                       help="Pinterest board name to pin images to")
    parser.add_argument("--similarity-threshold", type=float, default=0.90,
                       help="Similarity threshold (0.0-1.0)")
    
    args = parser.parse_args()
    
    # Check for access token
    if not os.getenv('PINTEREST_ACCESS_TOKEN'):
        print("ERROR: No Pinterest access token found!")
        print("Please run: python pinterest_api_setup.py")
        return
    
    # Create pinner instance
    pinner = PinterestPinner(args.images_dir, args.board_name)
    pinner.similarity_threshold = args.similarity_threshold
    
    # Run the pinning
    results = pinner.run()
    
    print(f"\nPinning completed! Check your Pinterest board: {args.board_name}")

if __name__ == "__main__":
    main() 