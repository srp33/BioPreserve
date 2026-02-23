#!/bin/bash

# --- Check Dependencies ---
for cmd in curl unzip; do
  command -v "$cmd" >/dev/null 2>&1 || { echo >&2 "$cmd is required but not installed. Aborting."; exit 1; }
done

# # --- Configuration ---
# ZIP_FILE="/tmp/HOMO_SAPIENS_6_1633734539.zip"
# EXTRACT_DIR="/data/gold/refine"
# URL="https://data/gold-refinery-s3-compendia-circleci-prod.s3.amazonaws.com/HOMO_SAPIENS_6_1633734539.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIAYSVVCNE5U2XKIEDW%2F20250617%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250617T170419Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEJH%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJGMEQCIEUUCPjPzvmueaUCseGdtIzUIGEqBbPyGBlJ10LAiGhGAiBDbdYPCAtKxbRzbWZ9SsZXr8hPRvfdXEVYy9WBrSMB%2Fyq5BQh5EAQaDDU4OTg2NDAwMzg5OSIMI5Nc0BNuSx8Dpg31KpYFc39eNvML5rAGY9KVq%2FIdAHtV532hZl8QsCocmnxG9CT25F7uzhlZYiKb%2Fe2EZMkKWgTot2qw8ijsnFzZGLILkqcGryb%2BL09hJ80h9fX71FNuP0gRuSjBWnRcS21l9n3cQ1eiPdT25cF0zhEnnTfI0b%2FE7cFy2AHXg0SQlnYWI06incQ%2F1VzTZPPeIILry9VqRrDyUXyH9Hp4jLTQ2itm3Lxo%2BafuqkSZNHN5CftONDiGog9hcytOk0vNlS7abYPa39hAwHo0Ux7cKDxTbKPjkQK%2FTDSi5iBucqo1do3X4fgX%2BoRuIPOXqTpF%2BJxErC1TtSV%2F0lyId1MOFiJr3jHmnNCWa8OWCBnmYwn6BYzLsjmyr7Oa9aMZ2P%2FN1k%2B3v560FZk2stMNuIhKYh5Vb8SVxcKEcL9tqMe73Iurb7YX%2FgzzJmuiQL96mnmhcyRNyTZDgwD5xMHkJ6PaKNYHMwDxa%2FRTn%2BD8Zr21RFyQR7vEcIDG0B4UAH%2B8bA4v3iDLe91N3UBoZQlDuMcDyHSZDvnftNpL3vot7MQftuttpq2K6pucZmY9Oq35%2FItPrzSlAv7fv2vAWrDUOguQDyaZtS4tq9omNXdcN2biZVQjyh77rys0T0SVayi15BgWLcUoNWg%2F32FyvwttjhjgtQnMcSuxT58uDzf5YzQ3kRP0Si0OpRHg32Q1CEvwAh5Wct%2FYCGZE2HqIL0Kp8AzhFg5GTJ%2FaGM1PXKycRVCUpaNHJAVm2ZCL6fxIsBQu0voabaEQK3uNXZ555HHBytk88ZJ0qNtc3ADJzb18Djlo06RhoRcl3j0Qnx0IbpmWkd02ewPvLYmEsnxQjagv%2BrIFSx6MWMJh5kyrVl6rRLmK19PFcWDZfr8Mu0Uv%2BRgwlqvGwgY6sgEpLVd4DBX6tS9g3zbLcPnxqfiuaRxf5FyTQmjaKB5JsxXVr0HmOH3Cnh7teBRPogX5CxVGSae3lhq%2BU5phs%2FqqRAPUZQrDR%2BHDYzw1u8y4a9v3pfLVHp5JnR8FSAQurMxnKZRujDRT0Qw15kzuoyxTzNDuSv0oueMdQIQ0g0sLwgqVVpGU62rz%2FU6eDEPz8t16JEv0pikoQIJ0xWRn82UPu8hk9k0b0IwfCy6mNG8U3Trb&X-Amz-Signature=2299aedd445d3e836272b93c87bc0314d984ba0e7a145c2dc1997b26834aa915"
# RETRIES=5

# # --- Helper: Fast ZIP Validation ---
# # Checks for the "End of Central Directory" record (PK\x05\x06), which is a
# # strong indicator that the file was not truncated during download.
# is_valid_zip() {
#   # The EOCD record is at the end, can be preceded by a comment < 66 Kb. 
#   tail -c 65536 "$1" | grep -q 'PK\x05\x06'
# }

# # --- Download File ---
# download() {
#   echo "Downloading $ZIP_FILE..."
#   if ! curl -L --fail --retry "$RETRIES" --retry-delay 5 -o "$ZIP_FILE" "$URL"; then
#     echo "Download failed after $RETRIES attempts." >&2
#     rm -f "$ZIP_FILE"
#     exit 1
#   fi
# }

# # --- Main ---
# if [ -f "$ZIP_FILE" ]; then
#   echo "Found existing $ZIP_FILE. Checking integrity..."
#   if ! is_valid_zip "$ZIP_FILE"; then
#     echo "Invalid or incomplete ZIP file. Removing and redownloading..."
#     rm -f "$ZIP_FILE"
#     download
#   else
#     echo "$ZIP_FILE is valid. Skipping download."
#   fi
# else
#   download
# fi

# # --- Unzip File ---
# echo "Unzipping $ZIP_FILE..."
# if ! unzip -o "$ZIP_FILE" -d "$EXTRACT_DIR"; then
#   echo "Error: Failed to unzip $ZIP_FILE." >&2
#   exit 1
# fi

echo "Extraction complete: $EXTRACT_DIR"

python /scripts/prepdata/convert_to_h5.py -i "/data/gold/refine" -o "/data/gold/refinebio.h5" --transpose
