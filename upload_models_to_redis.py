import redis

# Update these paths and credentials as needed
redis_host = "localhost"
redis_port = 6379
redis_password = "hqpl@123"

model1_path = "/Users/hqpl/Library/Caches/models/best_updated.pt"
model2_path = "/Users/hqpl/Library/Caches/models/best.pt"

r = redis.Redis(host=redis_host, port=redis_port, password=redis_password)

with open(model1_path, "rb") as f:
    r.set("model:best.pt", f.read())

with open(model2_path, "rb") as f:
    r.set("model:best_updated.pt", f.read())

print("Models uploaded to Redis.")