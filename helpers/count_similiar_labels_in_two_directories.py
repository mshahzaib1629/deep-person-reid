import os

def count_images_by_labels(folder_a, folder_b, labels):
    # Initialize dictionaries to store counts for each label in each folder
    counts_a = {label: 0 for label in labels}
    counts_b = {label: 0 for label in labels}

    # Get the list of files in folder A
    files_a = os.listdir(folder_a)
    # Get the list of files in folder B
    files_b = os.listdir(folder_b)

    # Iterate over the files in folder A and count the images for each label
    for file_name in files_a:
        for label in labels:
            if file_name.startswith(label):
                counts_a[label] += 1

    # Iterate over the files in folder B and count the images for each label
    for file_name in files_b:
        for label in labels:
            if file_name.startswith(label):
                counts_b[label] += 1

    return counts_a, counts_b

folder_a = "./reid-data/chunks/query/c11"
folder_b = "./reid-data/chunks/gallery/c11"
given_labels = ["0925", "0541", "0490", "0071", "0488", "0198", "1095", "0758", "0808", "1163", "0280", "0531", "0918", "0463", "0719", "1239", "0805", "0015", "0550", "0804", "0746", "1488", "0101", "1312", "1021", "0157", "1020", "1034", "1319", "1211", "1054", "1147", "0916", "0262", "1355", "0283", "0364", "0092", "1151", "0966", "0598", "1271", "0418", "1175", "0980", "1233", "0267", "1255", "1287", "0063", "0492", "0188", "0974", "0533", "1494", "0920", "1288", "0512", "0650", "0511", "1136", "0909", "0680", "1425", "0257", "1120", "0924", "1063", "0014", "1085", "0050", "0161", "0799", "1067", "1262", "0233", "0219", "0934", "0274", "0501", "0329", "0062", "0691", "1478", "0574", "0342", "0231", "0951", "0083", "1369", "0387", "1191", "0822", "1398", "0440", "1435", "1439", "0756", "0607", "0183", "1223", "0627", "0029", "1042", "1069", "0543", "0310", "0735", "0278", "0771", "0378", "0483", "0514", "0553", "1414", "1293", "1491", "1225", "1103", "0745", "0253", "0960", "1043", "1145", "1246", "0873", "0560", "0699", "1074", "1383", "1482", "1202", "0263", "0776", "0155", "1301", "1324", "1215", "1452", "1306", "0285", "0861", "0538", "0938", "0196", "0860", "1061", "0693", "0715", "0119", "0228", "1181", "0743", "0270", "0478", "1375", "0532", "0587", "1171", "0950", "0220", "1448", "0213", "0906", "0507", "1490", "1104", "0003", "0036", "1070", "0498", "0215", "0144", "1283", "0775", "0675", "0438", "0004", "1342", "0163", "0165", "1413", "0829", "0294", "1083", "0526", "1302", "1322", "1299", "0634", "1046", "0247", "0502", "0922", "0725", "0835", "0993", "1185", "0845", "1310", "0996", "0737", "0428", "0831", "0295", "0054", "0763", "0128", "0880", "1270", "1236", "0005", "1026", "0170", "1131", "0454", "0544", "0585", "0578", "0443", "0039", "1047", "0747", "1361", "0087",]

counts_in_folder_a, counts_in_folder_b = count_images_by_labels(folder_a, folder_b, given_labels)

for label in given_labels:
    print(f"\nLabel: {label}")
    print(f"Number of images in Folder A: {counts_in_folder_a[label]}")
    print(f"Number of images in Folder B: {counts_in_folder_b[label]}")
    print("----------------------------------------------------------------------------")
