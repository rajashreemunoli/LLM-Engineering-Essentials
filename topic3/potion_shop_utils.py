
import sqlite3
import os
import pandas as pd

def create_potion_shop_database(db_path='potion_shop.db'):
    """
    Create a new SQLite database with the potion shop schema and sample data.
    
    Args:
        db_path (str): Path where the database file will be saved
    
    Returns:
        sqlite3.Connection: Connection to the created database
    """
    # Check if database already exists
    if os.path.exists(db_path):
        print(f"Database already exists at {db_path}")
        return sqlite3.connect(db_path)
    
    # Create a new database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
    CREATE TABLE shop_inventory (
      potion_id INTEGER PRIMARY KEY,      -- Unique ID of the potion
      stock INTEGER,                      -- How many are in stock
      price INTEGER                       -- Price in gold
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE potions (
      potion_id INTEGER PRIMARY KEY,      -- Matches inventory ID
      potion_name TEXT,                   -- Name of the potion
      category TEXT,                      -- Category (healing, mana, etc.)
      effect TEXT,                        -- Specific effect (heals 10 hp, etc.)
      rarity TEXT,                        -- common, uncommon, rare, legendary
      duration TEXT,                      -- How long it lasts (e.g., '1 min', '10 min', 'permanent')
      side_effects TEXT                   -- Possible side effects (nullable)
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE purchases (
      purchase_id INTEGER PRIMARY KEY,
      customer_name TEXT,                 -- Name of the customer
      potion_id INTEGER,                  -- Purchased potion
      quantity INTEGER,                   -- Number bought
      date DATE,                          -- Date of purchase
      FOREIGN KEY(potion_id) REFERENCES potions(potion_id)
    )
    ''')
    
    # Insert sample data for potions
    potions_data = [
        (1, 'Minor Healing Potion', 'healing', 'heals 10 hp', 'common', 'instant', 'none'),
        (2, 'Healing Potion', 'healing', 'heals 25 hp', 'uncommon', 'instant', 'mild drowsiness'),
        (3, 'Major Healing Potion', 'healing', 'heals 50 hp', 'rare', 'instant', 'temporary weakness'),
        
        (4, 'Minor Mana Potion', 'mana', 'restores 15 mp', 'common', 'instant', 'none'),
        (5, 'Mana Potion', 'mana', 'restores 30 mp', 'uncommon', 'instant', 'mild headache'),
        (6, 'Major Mana Potion', 'mana', 'restores 60 mp', 'rare', 'instant', 'temporary brain fog'),
        
        (7, 'Crude Barkskin Potion', 'protection', 'reduces physical damage by 10%', 'common', '5 min', 'skin dryness'),
        (8, 'Refined Barkskin Potion', 'protection', 'reduces physical damage by 25%', 'rare', '10 min', 'reduced mobility'),
        
        (9, 'Swift Speed Potion', 'movement', 'increases movement speed by 30%', 'uncommon', '3 min', 'mild dizziness'),
        (10, 'Superior Speed Potion', 'movement', 'increases movement speed by 50%', 'rare', '5 min', 'exhaustion after effect'),
        
        (11, 'Frenzied Berserker Elixir', 'combat', 'increases attack damage by 25%', 'uncommon', '2 min', 'reduced defense'),
        (12, 'Mighty Berserker Elixir', 'combat', 'increases attack damage by 50%', 'rare', '3 min', 'tunnel vision'),
        
        (13, 'Clarity Tonic', 'mental', 'improves focus and perception', 'uncommon', '10 min', 'sensory overload'),
        (14, 'Dreamless Sleep Draught', 'rest', 'provides 8 hours of restful sleep in 4 hours', 'rare', '4 hours', 'grogginess upon waking'),
        (15, 'Antidote', 'curative', 'cures common poisons and toxins', 'uncommon', 'instant', 'stomach discomfort')
    ]
    cursor.executemany('INSERT INTO potions VALUES (?, ?, ?, ?, ?, ?, ?)', potions_data)
    
    # Insert sample data for inventory
    inventory_data = [
        (1, 25, 10),
        (2, 15, 25),
        (3, 5, 60),
        (4, 20, 8),
        (5, 12, 20),
        (6, 6, 45),
        (7, 10, 15),
        (8, 4, 40),
        (9, 8, 30),
        (10, 3, 65),
        (11, 7, 35),
        (12, 2, 70),
        (13, 5, 25),
        (14, 3, 50),
        (15, 10, 20)
    ]
    cursor.executemany('INSERT INTO shop_inventory VALUES (?, ?, ?)', inventory_data)
    
    # Insert sample data for purchases (fewer unique customers with multiple purchases)
    purchases_data = [
        (1, 'Alaric', 4, 3, '2024-01-15'),     # Mana potion
        (2, 'Elara', 5, 1, '2024-01-20'),      # Mana potion
        (3, 'Thorne', 1, 3, '2024-01-25'),     # Healing potion
        (4, 'Lyra', 7, 2, '2024-02-01'),       # Barkskin potion
        (5, 'Dorian', 3, 1, '2024-02-05'),     # Healing potion
        (6, 'Elara', 9, 1, '2024-02-10'),      # Speed potion
        (7, 'Thorne', 6, 2, '2024-02-15'),     # Mana potion
        (8, 'Alaric', 13, 1, '2024-02-20'),    # Clarity tonic
        (9, 'Lyra', 1, 2, '2024-02-25'),       # Healing potion
        (10, 'Dorian', 2, 3, '2024-03-01'),    # Healing potion
        (11, 'Thorne', 12, 1, '2024-03-05'),   # Berserker elixir
        (12, 'Alaric', 4, 2, '2024-03-10'),    # Mana potion
        (13, 'Seren', 15, 3, '2024-03-15'),    # Antidote
        (14, 'Seren', 3, 1, '2024-03-20'),     # Healing potion
        (15, 'Lyra', 10, 1, '2024-03-25'),     # Speed potion
        (16, 'Elara', 14, 1, '2024-04-01'),    # Sleep draught
        (17, 'Dorian', 8, 1, '2024-04-05'),    # Barkskin potion
        (18, 'Thorne', 11, 2, '2024-04-10'),   # Berserker elixir
        (19, 'Seren', 5, 3, '2024-04-15'),     # Mana potion
        (20, 'Alaric', 1, 5, '2024-04-20')     # Healing potion
    ]
    cursor.executemany('INSERT INTO purchases VALUES (?, ?, ?, ?, ?)', purchases_data)
    
    
    # Commit changes and close
    conn.commit()
    
    print(f"Database created successfully at {db_path}")
    return conn

def load_potion_shop_database(db_path='potion_shop.db'):
    """
    Load the potion shop database.
    
    Args:
        db_path (str): Path to the database file
    
    Returns:
        sqlite3.Connection: Connection to the database
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at {db_path}. Create it first with create_potion_shop_database()")
    
    # Connect to the database
    conn = sqlite3.connect(db_path)
    
    # Test the connection by listing tables
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    print("Available tables:")
    for table in tables:
        print(f" - {table[0]}")
    
    return conn

def query_db(conn, query, params=(), fetch_all=True):
    """
    Execute a query on the database.
    
    Args:
        conn (sqlite3.Connection): Connection to the database
        query (str): SQL query to execute
        params (tuple): Parameters for the query
        fetch_all (bool): Whether to fetch all results or just one
    
    Returns:
        list or tuple: Query results
    """
    cursor = conn.cursor()
    cursor.execute(query, params)
    
    if query.strip().upper().startswith(('SELECT', 'PRAGMA')):
        if fetch_all:
            return cursor.fetchall()
        else:
            return cursor.fetchone()
    else:
        conn.commit()
        return cursor.rowcount

def show_table(conn, table_name):
    """
    Display the contents of a table as a pandas DataFrame.
    
    Args:
        conn (sqlite3.Connection): Connection to the database
        table_name (str): Name of the table to display
    
    Returns:
        pandas.DataFrame: Table contents
    """
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql_query(query, conn)
    print(f"\nTable: {table_name}")
    print(df)
    return df

def get_table_schema(conn, table_name):
    """
    Get the schema of a table.
    
    Args:
        conn (sqlite3.Connection): Connection to the database
        table_name (str): Name of the table
    
    Returns:
        list: List of column information tuples
    """
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name});")
    return cursor.fetchall()

def show_schema(conn):
    """
    Display the schema of all tables in the database.
    
    Args:
        conn (sqlite3.Connection): Connection to the database
    """
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    print("\n== Database Schema ==")
    for table_name in [item[0] for item in tables]:
        schema = get_table_schema(conn, table_name)
        print(f"\nTable: {table_name}")
        for col in schema:
            pk_marker = " [PRIMARY KEY]" if col[5] == 1 else ""
            print(f"  {col[1]} ({col[2]}){pk_marker}")

