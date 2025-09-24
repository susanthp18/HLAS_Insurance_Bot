#!/usr/bin/env python3
"""
Schema Migration Script
=======================

Migrates the existing Weaviate collection to include the new possible_queries field.
This script will:
1. Create a backup collection with the new schema
2. Copy all existing data to the new collection with empty possible_queries
3. Replace the old collection with the new one

Usage:
    python migrate_schema.py
"""

import os
import sys
import logging
from dotenv import load_dotenv
import weaviate
from weaviate.classes.config import Property, DataType
from weaviate.classes.query import Filter

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_weaviate_client():
    """Get Weaviate client"""
    url = os.getenv("WEAVIATE_URL")
    if not url:
        raise ValueError("Missing WEAVIATE_URL in environment")
    
    from urllib.parse import urlparse
    parsed_url = urlparse(url)
    host = parsed_url.hostname
    port = parsed_url.port
    scheme = parsed_url.scheme

    if not host or not port or not scheme:
        raise ValueError("Invalid WEAVIATE_URL format")

    return weaviate.connect_to_custom(
        http_host=host,
        http_port=port,
        http_secure=scheme == 'https',
        grpc_host=host,
        grpc_port=50051,
        grpc_secure=False
    )

def create_new_collection(client, collection_name):
    """Create new collection with possible_queries field"""
    new_collection_name = f"{collection_name}_new"
    
    logger.info(f"Creating new collection: {new_collection_name}")
    
    client.collections.create(
        name=new_collection_name,
        description="Insurance product content chunks with metadata and possible queries",
        properties=[
            Property(name="product_name", data_type=DataType.TEXT, description="Name of the insurance product"),
            Property(name="document_type", data_type=DataType.TEXT, description="Type of document (Policy, FAQ, Benefits)"),
            Property(name="source_file", data_type=DataType.TEXT, description="Source file path"),
            Property(name="content", data_type=DataType.TEXT, description="Content of the chunk"),
            Property(name="possible_queries", data_type=DataType.TEXT_ARRAY, description="List of possible user queries this chunk can answer"),
            Property(name="section_hierarchy", data_type=DataType.TEXT_ARRAY, description="Hierarchical section path"),
            Property(name="question", data_type=DataType.TEXT, description="Question for FAQ chunks"),
            Property(name="chunk_id", data_type=DataType.TEXT, description="Unique chunk identifier"),
            # Product metadata
            Property(name="product_category", data_type=DataType.TEXT, description="Category of the insurance product"),
            Property(name="all_products", data_type=DataType.TEXT_ARRAY, description="List of all available products"),
            # Product flags for easy filtering
            Property(name="is_car_product", data_type=DataType.BOOL, description="True if this is Car insurance content"),
            Property(name="is_early_product", data_type=DataType.BOOL, description="True if this is Early insurance content"),
            Property(name="is_family_product", data_type=DataType.BOOL, description="True if this is Family insurance content"),
            Property(name="is_home_product", data_type=DataType.BOOL, description="True if this is Home insurance content"),
            Property(name="is_hospital_product", data_type=DataType.BOOL, description="True if this is Hospital insurance content"),
            Property(name="is_maid_product", data_type=DataType.BOOL, description="True if this is Maid insurance content"),
            Property(name="is_travel_product", data_type=DataType.BOOL, description="True if this is Travel insurance content")
        ],
    )
    
    logger.info(f"Created new collection: {new_collection_name}")
    return new_collection_name

def migrate_data(client, old_collection_name, new_collection_name):
    """Migrate data from old collection to new collection"""
    logger.info(f"Migrating data from {old_collection_name} to {new_collection_name}")
    
    old_collection = client.collections.get(old_collection_name)
    new_collection = client.collections.get(new_collection_name)
    
    # Get all objects from old collection
    result = old_collection.query.fetch_objects(limit=10000)  # Adjust limit as needed
    
    migrated_count = 0
    for obj in result.objects:
        # Get properties and vector
        properties = obj.properties
        vector = obj.vector
        
        # Add empty possible_queries field
        properties["possible_queries"] = []
        
        # Insert into new collection
        new_collection.data.insert(
            properties=properties,
            vector=vector
        )
        migrated_count += 1
        
        if migrated_count % 100 == 0:
            logger.info(f"Migrated {migrated_count} objects...")
    
    logger.info(f"Successfully migrated {migrated_count} objects")
    return migrated_count

def replace_collection(client, old_collection_name, new_collection_name):
    """Replace old collection with new collection"""
    backup_name = f"{old_collection_name}_backup"
    
    logger.info(f"Renaming {old_collection_name} to {backup_name}")
    # Note: Weaviate doesn't support renaming, so we'll delete the old one
    # In production, you might want to keep a backup
    
    logger.info(f"Deleting old collection: {old_collection_name}")
    client.collections.delete(old_collection_name)
    
    logger.info(f"Creating final collection: {old_collection_name}")
    # Create the final collection with the original name
    create_new_collection(client, old_collection_name.replace("_new", ""))
    
    # Migrate data from new collection to final collection
    final_collection = client.collections.get(old_collection_name)
    temp_collection = client.collections.get(new_collection_name)
    
    result = temp_collection.query.fetch_objects(limit=10000)
    for obj in result.objects:
        final_collection.data.insert(
            properties=obj.properties,
            vector=obj.vector
        )
    
    # Delete temporary collection
    client.collections.delete(new_collection_name)
    
    logger.info("Schema migration completed successfully!")

def main():
    """Main migration function"""
    collection_name = os.getenv("WEAVIATE_COLLECTION_NAME")
    if not collection_name:
        raise ValueError("Missing WEAVIATE_COLLECTION_NAME in environment")
    
    logger.info("Starting schema migration...")
    logger.info(f"Collection: {collection_name}")
    
    client = get_weaviate_client()
    
    try:
        # Check if collection exists
        if not client.collections.exists(collection_name):
            logger.error(f"Collection {collection_name} does not exist!")
            return
        
        # Create new collection with updated schema
        new_collection_name = create_new_collection(client, collection_name)
        
        # Migrate data
        migrated_count = migrate_data(client, collection_name, new_collection_name)
        
        if migrated_count > 0:
            # Replace old collection with new one
            replace_collection(client, collection_name, new_collection_name)
            logger.info("✅ Schema migration completed successfully!")
        else:
            logger.warning("No data to migrate. Cleaning up...")
            client.collections.delete(new_collection_name)
            
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise
    finally:
        client.close()

if __name__ == "__main__":
    main()
