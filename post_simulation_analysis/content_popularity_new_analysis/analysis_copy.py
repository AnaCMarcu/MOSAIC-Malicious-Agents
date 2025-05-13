import pandas as pd
import sqlite3

import sqlite3
import json


def get_db_schema_as_sql(db_name: str):
    """
    Extract the schema from a SQLite database as CREATE TABLE SQL statements.
    """
    # Connect to the SQLite database
    conn = sqlite3.connect(db_name)

    # Create a cursor object
    cursor = conn.cursor()

    # Get all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    schema_sql = {}

    for table in tables:
        table_name = table[0]
        # Get the CREATE TABLE statement for this table
        cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}';")
        create_statement = cursor.fetchone()[0]
        schema_sql[table_name] = create_statement

    # Close the connection
    conn.close()

    return schema_sql



def export_db_table_to_csv(db_name: str, table_name: str):
    # Connect to the SQLite database
    # Replace 'path_to_your_db.db' with your actual database file path
    conn = sqlite3.connect(db_name)

    # Query the database and load into a pandas DataFrame
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql_query(query, conn)

    # Export to CSV
    df.to_csv(f'{table_name}.csv', index=False)

    # Close the connection
    conn.close()

if __name__ == '__main__':

    # # Example usage
    # db_schema_sql = get_db_schema_as_sql('simulation.db')
    #
    # # Print the SQL statements
    # for table_name, create_statement in db_schema_sql.items():
    #     print(f"-- Table: {table_name}")
    #     print(f"{create_statement};")
    #     print()
    #
    # # Save the SQL schema to a file
    # with open('db_schema.sql', 'w') as f:
    #     for table_name, create_statement in db_schema_sql.items():
    #         f.write(f"-- Table: {table_name}\n")
    #         f.write(f"{create_statement};\n\n")
    #
    # # Also save as JSON if needed
    # with open('db_schema.json', 'w') as f:
    #     json.dump(db_schema_sql, f, indent=4)
    export_db_table_to_csv('20250512_225451.db', 'follows')
    export_db_table_to_csv('20250512_225451.db', 'users')
    export_db_table_to_csv('20250512_225451.db', 'posts')
    export_db_table_to_csv('20250512_225451.db', 'comments')
    export_db_table_to_csv('20250512_225451.db', 'user_actions')

    # export_db_table_to_csv('20250512_225451.db', 'follows')
    # export_db_table_to_csv('20250122_215428.db', 'posts')
    # export_db_table_to_csv('20250122_215428.db', 'users')
    # export_db_table_to_csv('20250122_215428.db', 'comments')
    # export_db_table_to_csv('20250122_215428.db', 'user_actions')