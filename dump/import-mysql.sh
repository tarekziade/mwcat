#!/bin/bash

set -x

SQL_FILES_DIR="./data"
DB_VOLUME_NAME="./db"
CONTAINER_NAME="mysql"
DB_NAME="wikipedia"
DB_USER="wikipedia"
DB_PASS="wikipedia"
DB_ROOT_PASSWORD="root"
DB_PORT="3306"

mkdir -p $DB_VOLUME_NAME

# Check if data is already imported
if [ -f "$SQL_FILES_DIR/.imported" ]; then
	echo "Data already imported. Skipping import."
else
	# Run the Docker container
	docker run -d \
		--name $CONTAINER_NAME \
		-v $SQL_FILES_DIR:/data \
		-v $DB_VOLUME_NAME:/var/lib/mysql \
		-e MYSQL_ROOT_PASSWORD=$DB_ROOT_PASSWORD \
		-e MYSQL_DATABASE=$DB_NAME \
		-e MYSQL_USER=$DB_USER \
		-e MYSQL_PASSWORD=$DB_PASS \
		-p $DB_PORT:3306 \
		mysql \
		--innodb-buffer-pool-size=4G \
		--innodb-log-buffer-size=256M \
		--innodb-log-file-size=1G \
		--innodb-write-io-threads=16 \
		--innodb-flush-log-at-trx-commit=0

	docker exec -i $CONTAINER_NAME /usr/bin/microdnf install -y epel-release
	docker exec -i $CONTAINER_NAME /usr/bin/microdnf install -y pv

	sleep 30

	# Import the SQL files
	#echo "Importing SQL files..."
	#for file in $SQL_FILES_DIR/*.sql; do
	#		docker exec -i $CONTAINER_NAME pv /$file | mysql -h 127.0.0.1 -u$DB_USER -p$DB_PASS $DB_NAME
	#done
	#docker exec -i $CONTAINER_NAME pv /data/enwiki-latest-pagelinks.sql | mysql -h 127.0.0.0.1 -u$DB_USER -p$DB_PASS $DB_NAME

	# Mark as imported
	touch "$SQL_FILES_DIR/.imported"
	echo "Import completed."
fi
