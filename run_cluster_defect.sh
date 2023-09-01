docker stop cluster_defect_search_engine
docker rm cluster_defect_search_engine
docker run --name cluster_defect_search_engine -p 8502:8502 -itd cluster_defect_search_engine
