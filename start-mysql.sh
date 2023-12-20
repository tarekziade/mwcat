#!/bin/bash

# Set these variables according to your setup
SQL_FILES_DIR="/path/to/sql/files"
DB_VOLUME_NAME="my_db_volume"
CONTAINER_NAME="my_mysql_container"
DB_NAME="mydatabase"
DB_USER="root"
DB_PASS="password"
DB_ROOT_PASSWORD="rootpassword"
DB_PORT="3306"

# Check if data is already imported
if [ -f "$SQL_FILES_DIR/.imported" ]; then
	echo "Data already imported. Skipping import."
else
	# Run the Docker container
	docker run -d \
		--name $CONTAINER_NAME \
		-v $DB_VOLUME_NAME:/var/lib/mysql \
		-v $SQL_FILES_DIR:/docker-entrypoint-initdb.d \
		-e MYSQL_ROOT_PASSWORD=$DB_ROOT_PASSWORD \
		-e MYSQL_DATABASE=$DB_NAME \
		-e MYSQL_USER=$DB_USER \
		-e MYSQL_PASSWORD=$DB_PASS \
		-p $DB_PORT:3306 \
		mysql

	# Wait for the database to start up
	echo "Waiting for database to start up..."
	sleep 30

	# Import the SQL files
	echo "Importing SQL files..."
	for file in $SQL_FILES_DIR/*.sql; do
		docker exec -i $CONTAINER_NAME mysql -u$DB_USER -p$DB_PASS $DB_NAME <"$file"
	done

	# Mark as imported
	touch "$SQL_FILES_DIR/.imported"
	echo "Import completed."
fi
