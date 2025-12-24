#!/usr/bin/env python3
"""
Migration script to drop obsolete image-related columns from the guild_config table.

This script removes columns that were part of the image generation feature which 
has since been removed from the codebase:
- image_window_seconds
- image_max_per_window
- user_daily_image_limit
- global_daily_image_limit

Usage:
    # Connect via SSH tunnel and run
    ssh yourserver 'python3 /path/to/migrate_drop_image_columns.py /data/chad.sqlite3'
    
    # Or copy the script and run it directly on the server
    scp migrate_drop_image_columns.py yourserver:/tmp/
    ssh yourserver 'python3 /tmp/migrate_drop_image_columns.py /data/chad.sqlite3'
"""

import sqlite3
import sys
from pathlib import Path


# Columns to remove from guild_config
OBSOLETE_COLUMNS = [
    "image_window_seconds",
    "image_max_per_window", 
    "user_daily_image_limit",
    "global_daily_image_limit",
]


def get_existing_columns(cursor: sqlite3.Cursor, table: str) -> list[str]:
    """Get list of existing column names in a table."""
    cursor.execute(f"PRAGMA table_info({table});")
    return [row[1] for row in cursor.fetchall()]


def get_sqlite_version(cursor: sqlite3.Cursor) -> tuple[int, int, int]:
    """Get SQLite version as a tuple (major, minor, patch)."""
    cursor.execute("SELECT sqlite_version();")
    version_str = cursor.fetchone()[0]
    parts = version_str.split(".")
    return tuple(int(p) for p in parts[:3])


def drop_column_modern(conn: sqlite3.Connection, table: str, column: str) -> None:
    """Drop column using ALTER TABLE DROP COLUMN (SQLite 3.35+)."""
    conn.execute(f"ALTER TABLE {table} DROP COLUMN {column};")
    conn.commit()


def drop_columns_legacy(
    conn: sqlite3.Connection, table: str, columns_to_drop: list[str]
) -> None:
    """
    Drop columns using table recreation (for SQLite < 3.35).
    
    This creates a new table without the unwanted columns, copies data,
    drops the old table, and renames the new one.
    """
    cursor = conn.cursor()
    
    # Get current schema
    cursor.execute(f"PRAGMA table_info({table});")
    all_columns_info = cursor.fetchall()
    
    # Filter out columns to drop
    columns_to_keep = [
        (name, col_type, notnull, default, pk)
        for cid, name, col_type, notnull, default, pk in all_columns_info
        if name not in columns_to_drop
    ]
    
    if not columns_to_keep:
        raise ValueError("Cannot drop all columns from table")
    
    # Build column definitions for new table
    column_defs = []
    for name, col_type, notnull, default, pk in columns_to_keep:
        parts = [name, col_type]
        if pk:
            parts.append("PRIMARY KEY")
        if notnull and not pk:
            parts.append("NOT NULL")
        if default is not None:
            parts.append(f"DEFAULT {default}")
        column_defs.append(" ".join(parts))
    
    column_names = [c[0] for c in columns_to_keep]
    column_list = ", ".join(column_names)
    
    # Create new table, copy data, swap
    conn.execute("BEGIN TRANSACTION;")
    try:
        conn.execute(
            f"CREATE TABLE {table}_new ({', '.join(column_defs)});"
        )
        conn.execute(
            f"INSERT INTO {table}_new ({column_list}) SELECT {column_list} FROM {table};"
        )
        conn.execute(f"DROP TABLE {table};")
        conn.execute(f"ALTER TABLE {table}_new RENAME TO {table};")
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def main():
    if len(sys.argv) != 2:
        print("Usage: python migrate_drop_image_columns.py <path_to_database>")
        print("Example: python migrate_drop_image_columns.py /data/chad.sqlite3")
        sys.exit(1)
    
    db_path = Path(sys.argv[1])
    if not db_path.exists():
        print(f"Error: Database file not found: {db_path}")
        sys.exit(1)
    
    print(f"Connecting to database: {db_path}")
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Check SQLite version
    version = get_sqlite_version(cursor)
    print(f"SQLite version: {'.'.join(str(v) for v in version)}")
    supports_drop_column = version >= (3, 35, 0)
    
    # Get existing columns
    existing = get_existing_columns(cursor, "guild_config")
    print(f"Current guild_config columns: {existing}")
    
    # Find which obsolete columns actually exist
    columns_to_remove = [col for col in OBSOLETE_COLUMNS if col in existing]
    
    if not columns_to_remove:
        print("✓ No obsolete image columns found. Database is already up to date.")
        conn.close()
        return
    
    print(f"Found obsolete columns to remove: {columns_to_remove}")
    
    # Backup reminder
    print("\n⚠️  This will modify the database. Make sure you have a backup!")
    response = input("Continue? [y/N]: ").strip().lower()
    if response != "y":
        print("Aborted.")
        conn.close()
        sys.exit(0)
    
    # Remove columns
    if supports_drop_column:
        print("Using modern ALTER TABLE DROP COLUMN...")
        for col in columns_to_remove:
            print(f"  Dropping column: {col}")
            drop_column_modern(conn, "guild_config", col)
    else:
        print("Using legacy table recreation method...")
        drop_columns_legacy(conn, "guild_config", columns_to_remove)
    
    # Verify
    remaining = get_existing_columns(cursor, "guild_config")
    print(f"\n✓ Migration complete!")
    print(f"  Remaining columns: {remaining}")
    
    # Check for any removed columns that might still exist
    leftover = [col for col in OBSOLETE_COLUMNS if col in remaining]
    if leftover:
        print(f"  ⚠️  Warning: Some columns still present: {leftover}")
    else:
        print("  ✓ All obsolete image columns removed successfully.")
    
    conn.close()
    print("\nDone. Restart your application to verify the fix.")


if __name__ == "__main__":
    main()
