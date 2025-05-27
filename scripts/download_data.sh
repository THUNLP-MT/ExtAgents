mkdir ./data
cd ./data

wget "https://drive.usercontent.google.com/download?id=1GGuInVLQ8SohdUTQiyj0t91Y_u8H-ZzN&export=download&authuser=0&confirm=t&uuid=88a8fa55-73d7-4449-b9bd-34a2e8e511e3&at=APcmpozXVD94SpWkcKaBMnkGP0Ud:1746353396463" -O sampled_hotpot_questions.json
# wget "https://drive.usercontent.google.com/download?id=1MWBchJnwA1KHWILei8Php58q8qPudqsi&export=download&authuser=0&confirm=t&uuid=dd5a8b1e-51e2-4384-95a6-5a1efe509916&at=APcmpozXOcj1aqixGGxQfinrPc-I:1746353388941" -O rag_128k.jsonl
wget "https://drive.usercontent.google.com/download?id=1Vm1nVdHeUZ_2GHgC0x6AvGa5IbcLzawz&export=download&authuser=0&confirm=t&uuid=4dee9ab8-206b-494e-aae6-030b1c250387&at=APcmpozDsCCtQScgtOqmoVIz9a7u:1746353391211" -O rag_1000k.jsonl
wget "https://drive.usercontent.google.com/download?id=1lkswqMzUGf3HPj4VCG1UifJQfqYievBf&export=download&authuser=0&confirm=t&uuid=23f58d70-d7a7-470a-a7fa-cc0f93a6f3e3&at=APcmpoz5LHYBeCFSuIM_Gvb7mdZ9:1746353372536" -O longbook_qa_eng.jsonl
wget "https://drive.usercontent.google.com/download?id=1FwG28KuHM9Ay11q_4QsnirB1SMxZbTJ1&export=download&authuser=0&confirm=t&uuid=7f824c4c-1fad-445e-a3f9-962c682974c4&at=APcmpoyN7SAEHwqo7F8klvifAYut:1746353370218" -O longbook_qa_chn.jsonl

# if you cannot access the google drive, you can use the following links to download the data

# wget "https://cloud.tsinghua.edu.cn/seafhttp/files/cc31dd4a-d6b2-4fcc-a897-e72cb6d76ac3/sampled_hotpot_questions.json"
# # wget "https://cloud.tsinghua.edu.cn/seafhttp/files/c85665ea-934f-4e91-a855-ef29193c66d4/rag_128k.jsonl"
# wget "https://cloud.tsinghua.edu.cn/seafhttp/files/bba91aef-963a-4ebd-9208-920f4a57ba85/rag_1000k.jsonl"
# wget "https://cloud.tsinghua.edu.cn/seafhttp/files/2021e89a-61f5-4fb5-9173-2a6161ea6827/longbook_qa_eng.jsonl"
# wget "https://cloud.tsinghua.edu.cn/seafhttp/files/32e22e18-2b64-4a46-9fbf-252bf676f41f/longbook_qa_chn.jsonl"