
import annoy
import pymongo

# Create the Annoy index
index = annoy.AnnoyIndex(128, 'angular')
for i in range(1000):
    v = [random.gauss(0, 1) for z in range(128)]
    index.add_item(i, v)

index.build(10)

# Convert the index to binary data
index_binary = index.tobinary()

# Connect to MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["mydatabase"]
collection = db["mycollection"]

# Store the binary data in MongoDB
document = {"index": index_binary}
collection.insert_one(document)

# Load the index from MongoDB
result = collection.find_one()
index_binary = result["index"]

# Convert the binary data back to an Annoy index
index = annoy.AnnoyIndex(128, 'angular')
index.load(index_binary)

