#!/bin/bash

# Folder where backups will be stored
BACKUP_DIR="./backup"
mkdir -p "$BACKUP_DIR"

# Get latest backup if exists
LATEST_BACKUP=$(ls -t "$BACKUP_DIR"/*.tar.gz 2>/dev/null | head -n1)
BACKUP_NAME="backup_$(date +%Y%m%d_%H%M%S).tar.gz"

if [ -f "$LATEST_BACKUP" ]; then
    SIZE=$(du -h "$LATEST_BACKUP" | cut -f1)
    echo "🗃️  Latest backup: $(basename "$LATEST_BACKUP")"
    echo "📦 Size: $SIZE"
    echo
    echo "Choose an option:"
    echo "1) Restore from latest backup"
    echo "2) Create new backup"
    echo "3) Cancel"
    read -p "Enter your choice [1-3]: " CHOICE

    case "$CHOICE" in
        1)
            echo "⚠️  This will restore the backup and overwrite current files."
            read -p "Are you sure? (y/n): " CONFIRM
            if [[ "$CONFIRM" == "y" || "$CONFIRM" == "Y" ]]; then
                echo "🕓 Restoring from $LATEST_BACKUP..."
                tar -xvzf "$LATEST_BACKUP" -C .
                echo "✅ Restore complete."
            else
                echo "❌ Restore canceled."
            fi
            ;;
        2)
            echo "📦 Creating new backup..."
            tar --exclude="./backup" -czvf "$BACKUP_DIR/$BACKUP_NAME" .
            echo "✅ Backup created: $BACKUP_NAME"
            ;;
        3)
            echo "❌ Operation canceled."
            ;;
        *)
            echo "❌ Invalid choice."
            ;;
    esac
else
    echo "📁 No backup found yet."
    echo "📦 Creating first backup..."
    tar --exclude="./backup" -czvf "$BACKUP_DIR/$BACKUP_NAME" .
    echo "✅ Backup created: $BACKUP_NAME"
fi
