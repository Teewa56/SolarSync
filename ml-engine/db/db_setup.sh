#!/bin/bash

# Database setup script for SolarSync

echo "Setting up SolarSync database..."

# Read database configuration
DB_NAME="solarsync"
DB_USER="solarsync_user"
DB_PASSWORD="solarsync_password"

# Create database and user
sudo -u postgres psql -c "CREATE DATABASE $DB_NAME;"
sudo -u postgres psql -c "CREATE USER $DB_USER WITH PASSWORD '$DB_PASSWORD';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $DB_USER;"

# Run SQL files
echo "Creating schema..."
psql -d $DB_NAME -U $DB_USER -f database_schema.sql

echo "Loading sample data..."
psql -d $DB_NAME -U $DB_USER -f sample_data.sql

echo "Creating views and functions..."
psql -d $DB_NAME -U $DB_USER -f views_and_functions.sql

echo "Database setup completed!"