import json
"""데이터를 만드는 코드입니다."""
# Load the data from the JSON file
with open('/workspace/jun4090/project/ai말평/일상대화요약_test.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Initialize the new dataset list
new_dataset_adjusted = []

# Adjust the speaker numbering to start from "화자 1" for each conversation
for item in data:
    new_item = {
        "prompt": "다음 대화를 요약해주세요.",
        "input": "",
        "output": ""
    }
    
    # Construct the input field with speaker differentiation starting from "화자 1"
    conversation = item["input"]["conversation"]
    speaker_mapping = {}
    current_speaker_number = 1
    conversation_text = ""
    
    for turn in conversation:
        speaker = turn["speaker"]
        if speaker not in speaker_mapping:
            speaker_mapping[speaker] = f"화자 {current_speaker_number}"
            current_speaker_number += 1
        speaker_id = speaker_mapping[speaker]
        utterance = turn["utterance"]
        conversation_text += f"{speaker_id}: {utterance}\n"
    
    new_item["input"] = conversation_text.strip()
    
    # Replace speaker ids in the output summary
    output_text = item["output"]
    for original_speaker, mapped_speaker in speaker_mapping.items():
        output_text = output_text.replace(original_speaker, mapped_speaker)
    new_item["output"] = output_text
    
    new_dataset_adjusted.append(new_item)

# Save the adjusted new dataset to a JSON file
new_dataset_adjusted_path = '/workspace/jun4090/project/ai말평/test_data.json'
with open(new_dataset_adjusted_path, 'w', encoding='utf-8') as new_file:
    json.dump(new_dataset_adjusted, new_file, ensure_ascii=False, indent=4)

print(new_dataset_adjusted_path)
