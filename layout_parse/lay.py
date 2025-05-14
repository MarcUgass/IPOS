import layoutparser as lp
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json

image = cv2.imread("./image2.jpeg")
if image is None:
    print("Error: No se pudo cargar la imagen. Verifica la ruta.")
else:
    image_rgb = image[..., ::-1].copy() # Convertir a RGB y crear una copia    
    print("Cargando el modelo Detectron2...")

    #COMPROBAR VERSIO PILOW, HA DE SER 8.4

    model = lp.models.detectron2.Detectron2LayoutModel(
            'lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
            label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"},
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.2] # 0.8 = umbral de confianza
        )
    
    layout = model.detect(image_rgb)

    image_with_boxes = lp.draw_box(image_rgb, layout, box_width=3)
    

    # Guardar la imagen modificada con OpenCV
    plt.imshow(image_with_boxes)
    plt.title("Imagen con Layout (Dibujo directo con OpenCV)")
    plt.axis('off')
    plt.show()

    text_blocks = lp.Layout([b for b in layout if (b.type=='Text' or b.type=='Title') ])
    figure_blocks = lp.Layout([b for b in layout if b.type=='Figure'])
    
    form = []
    
    ocr_agent = lp.TesseractAgent(languages='eng')


    for block in text_blocks:
        x1 = int(block.block.x_1)
        y1 = int(block.block.y_1)
        x2 = int(block.block.x_2)
        y2 = int(block.block.y_2)
        block_type = block.type

        segment_image = (block
                        .pad(left=5, right=5, top=5, bottom=5) #padding
                        .crop_image(image))
        
        text = ocr_agent.detect(segment_image)

        #block.set(text=text, inplace=True) #settejarho al mateix out
        form.append({
        "x1": (x1+5)/2,
        "y1": (y1-5)/2,
        "x2": (x2+5)/2,
        "y2": y2/1.2,
        "text": text,
        "type": block_type
    })
    
    for i, item in enumerate(form, start=1):
        print(f"Bloque {i}:")
        print(f"  x1: {item['x1']}, y1: {item['y1']}, x2: {item['x2']}, y2: {item['y2']}")
        print(f"  text: {item['text']}")
        print(f"  type: {item['type']}")
        print("-" * 40)

    #form_pdf = []
    """
    IMG_WIDTH_PX, IMG_HEIGHT_PX = image.size
    dpi = image.info.get('dpi', (72, 72)) 
    IMG_PPI = dpi[0] if dpi[0] > 0 else 72.0 
    """

    converted_form = {}


    
    for idx, item in enumerate(form):
        converted_form[str(idx)] = {
            "type": "text",
            "text": item["text"],
            "font": "Times-Roman",
            "size": 8,
            "color": [0, 0, 0],
            "position": [item["x1"], item["y1"], item["x2"], item["y2"]],
            "align": "left",
            "valign": "top"
        }
    
    # Guardar en archivo JSON
    with open("form_pdf_layout.json", "w", encoding="utf-8") as f:
        json.dump(converted_form, f, ensure_ascii=False, indent=4)



"""
    image_with_boxes = lp.draw_box(image_rgb, layout, box_width=3)
    

    # Guardar la imagen modificada con OpenCV
    plt.imshow(image_with_boxes)
    plt.title("Imagen con Layout (Dibujo directo con OpenCV)")
    plt.axis('off')
    plt.show()
"""