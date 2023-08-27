from bs4 import BeautifulSoup

def extract_ids_and_image_names(html):
    soup = BeautifulSoup(html, 'html.parser')
    div_blocks = soup.find_all('div')
    id_image_info = []

    for div_block in div_blocks:
        img_tag = div_block.find('img')
        if img_tag:
            div_id = div_block.get('id')
            img_src = img_tag.get('src')
            if div_id and img_src:
                id_image_info.append({'id': div_id, 'img_src': img_src})

    return id_image_info

def extract_ids_and_image_links(html):
    soup = BeautifulSoup(html, 'html.parser')
    div_blocks = soup.find_all('div')
    id_link_info = []

    for div_block in div_blocks:
        link_tag = div_block.find('a', class_='photo_wrap')
        if link_tag:
            div_id = div_block.get('id')
            link_href = link_tag.get('href')
            if div_id and link_href:
                id_link_info.append({'id': div_id, 'link_href': link_href})

    return id_link_info

for i in range(1,18):
    # Read HTML content from a file
    file_path = ''
    if i == 1:
        file_path = './html_msg_data/messages.html'
    else :
        file_path = f'./html_msg_data/messages{i}.html'

    with open(file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()

    # Extract ids and image names
    result = extract_ids_and_image_links(html_content)

    output_file = './grbrt_spb/imgs_data.txt'
    with open(output_file, 'a', encoding='utf-8') as output:
        for entry in result:
            output.write(f'{entry}\n')
