import sqlite3

db_path = 'mlflow.db'
old_path = 'C:/Users/rafal/OneDrive/Dokumente/GitHub/Movie-Recommendation-System-with-Explainability'
new_path = 'C:/Users/Admin/OneDrive/Desktop/machine_learning_projekt/newer/Movie-Recommendation-System-with-Explainability'

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Update experiments table
cursor.execute("SELECT experiment_id, artifact_location FROM experiments")
for row in cursor.fetchall():
    exp_id, loc = row
    if loc and old_path in loc:
        new_loc = loc.replace(old_path, new_path)
        cursor.execute("UPDATE experiments SET artifact_location = ? WHERE experiment_id = ?", (new_loc, exp_id))

# Update runs table
cursor.execute("SELECT run_uuid, artifact_uri FROM runs")
for row in cursor.fetchall():
    run_id, loc = row
    if loc and old_path in loc:
        new_loc = loc.replace(old_path, new_path)
        cursor.execute("UPDATE runs SET artifact_uri = ? WHERE run_uuid = ?", (new_loc, run_id))

conn.commit()
conn.close()
print("Paths updated successfully.")
