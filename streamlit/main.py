import os
import numpy as np
import pandas as pd
from skimage import io
import streamlit as st
import altair as alt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import torch
import albumentations
import matplotlib.pyplot as plt
import copy
import re
from sklearn import metrics
import plotly.express as px

from models import ExtraSmallUNet, apply_normalization, apply_augmentations, map_mask_classes

plt.rcParams['savefig.pad_inches'] = "0"
font = {'family': 'Sans',
        'weight': 'normal',
        'size': 6.5}
# https://matplotlib.org/3.1.1/tutorials/text/text_props.html
plt.rc('font', **font)

CLASSES = [
    "Alcantarillado", "Marca vial", "Hueco", "Parcheo", "Grietas longitudinales",
    "Grietas en forma de piel de cocodrilo", "Grietas transversales",
    "Meteorización y desprendimiento",  # "Espiras Magnéticas"
]

SHORT_CLASSES = {
    "Grietas en forma de piel de cocodrilo": "Cocodrilo",
    "Meteorización y desprendimiento": "Meteorización",
    "Grietas longitudinales": "G. longitudinales",
    "Grietas transversales": "G. transversales",
}

SEGMENTATION_CLASSES = {"Parcheo": 1, "Hueco": 2, "Grietas longitudinales": 3, "Grietas transversales": 4}

PLOTS2AX = {2: (1, 2), 3: (1, 3), 4: (2, 2)}
AX2POSITIONS = {(1, 2): [0, 1], (1, 3): [0, 1, 2], (2, 2): [(0, 0), (0, 1), (1, 0), (1, 1)]}

NoneFold = "NoneFold"


def main():
    st.set_page_config(page_title="Análisis SIMEPU")

    hide_streamlit_style()

    st.sidebar.title("¿Qué hacer?")
    app_mode = st.sidebar.selectbox(
        "Selecciona un modo",
        ["Información General", "Metadatos", "Análisis redes"]
    )

    st.title("SIMEPU")

    if app_mode == "Información General":
        # Render the readme as markdown using st.markdown.
        st.markdown(get_file_content_as_string("info_general.md"))
    elif app_mode == "Metadatos":
        metadatos()
    elif app_mode == "Análisis redes":
        redes()


def hide_streamlit_style():
    return st.markdown(
        """<style>#MainMenu {visibility: hidden;}
        footer {visibility: hidden;}</style>""",
        unsafe_allow_html=True
    )


@st.cache(show_spinner=False)
def get_file_content_as_string(path):
    file1 = open(path, 'r')
    lines = file1.readlines()
    res = ""
    for line in lines:
        res += line.strip() + "\n"
    return res


def metadatos():
    df_truth = load_full_df("truth")

    st.markdown("""
    En la siguiente página mostramos información acerca de la base de datos y su etiquetado
    """)
    st.write(f"Número total de imágenes: `{len(df_truth)}`.")

    num_classes_imagenes = df_truth.rename(columns={"file_path": "# Imágenes", "sum": "# Clases"})
    clases_num_imagenes = pd.DataFrame(
        df_truth[np.intersect1d(df_truth.columns, CLASSES)].sum(), columns=["# Imágenes"]
    ).reset_index()
    clases_num_imagenes = clases_num_imagenes.rename(columns={"index": "Clases"})
    clases_num_imagenes.Clases.replace({k: v for k, v in SHORT_CLASSES.items()}, inplace=True)

    # No es 'necesario' width ya que indicamos mas abajo que utilice el ancho del contenedor
    cni = alt.Chart(clases_num_imagenes).mark_bar().encode(
        alt.X('Clases:O', axis=alt.Axis(labelAngle=-45, labelFontSize=12)),
        alt.Y('# Imágenes:Q'),
        tooltip=['Clases', '# Imágenes']
    ).properties(height=500, title="Número de imágenes por clase")

    nci = alt.Chart(num_classes_imagenes.groupby("# Clases").count().reset_index()).mark_bar().encode(
        alt.X('# Clases:O', axis=alt.Axis(values=np.arange(df_truth["sum"].max() + 1), labelAngle=0, labelFontSize=12)),
        alt.Y('# Imágenes:Q'),
        tooltip=['# Clases', '# Imágenes']
    ).properties(height=400, title="Número de imágenes por número de clases en la imagen")

    st.altair_chart(cni, use_container_width=True)
    st.altair_chart(nci, use_container_width=True)

    masks_metadata = count_masks()
    lm_df = pd.DataFrame(masks_metadata.items(), columns=["Clases", "# Imágenes"])
    lm_df.Clases.replace({k: v for k, v in SHORT_CLASSES.items()}, inplace=True)
    lm = alt.Chart(lm_df).mark_bar().encode(
        alt.X('Clases:O', axis=alt.Axis(labelAngle=-45, labelFontSize=12)),
        alt.Y('# Imágenes:Q'),
        tooltip=['Clases', '# Imágenes']
    ).properties(height=500, title="Número de imágenes segmentadas por clase")
    st.altair_chart(lm, use_container_width=True)

    st.markdown("## Resultados")
    st.markdown("### Clasificación Multi-Etiqueta")
    targets, logits = load_multilabel_predictions()
    umbral = st.number_input('Selecciona un umbral', min_value=0.0, max_value=1.0001, value=0.5, step=0.05)
    logits_thresholded = (logits > umbral).astype(np.uint8)
    avg = "micro"
    res = {"Clase": [], "Accuracy": [], "Recall": [], "Precision": [], "F1": []}
    for indx, clase in enumerate(CLASSES):
        pr = precision_score(targets[:, indx], logits_thresholded[:, indx], average=avg)
        re = recall_score(targets[:, indx], logits_thresholded[:, indx], average=avg)
        f1 = f1_score(targets[:, indx], logits_thresholded[:, indx], average=avg)
        acc = accuracy_score(targets[:, indx], logits_thresholded[:, indx])
        res["Clase"].append(clase)
        res["Accuracy"].append(acc)
        res["Recall"].append(re)
        res["Precision"].append(pr)
        res["F1"].append(f1)
    st.table(pd.DataFrame(res))

    # -- ROC 1 --
    st.write("### Curva ROC v1")
    st.write(""" - Ratio Verdaderos Positivos: Aquellos casos con daño, observando el *ground truth*, donde la
            predicción y *ground truth* es es igual *atómicamente*. Por ejemplo, p=[0,1,0], gt=[0,1,1] tendríamos, 
            mirando únicamente los casos con daño (1s), un caso correcto, mientras que p=[0,1,0], gt=[0,1,0] también 
            tiene un caso correcto. Se suman los casos correctos y se divide entre el número de casos con daño (1s).""")
    st.write(""" - Ratio Falsos Positivos: Aquellos casos sin daño (0s), observando el *ground truth*, cuya predicción 
            es de daño (1). Se suman estos casos y se divide entre el número de casos sin daño. Se realiza de forma
            *atómica*.""")
    y = targets.ravel()
    scores = logits.ravel()
    fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=1)

    fig_roc1 = px.line(
        x=fpr, y=tpr, labels={
            "x": "Ratio Falsos Positivos",
            "y": "Ratio Verdaderos Positivos",
        },
    )

    st.plotly_chart(fig_roc1, use_container_width=True)

    # -- ROC 2 --
    st.write("### Curva ROC v2")
    st.write(""" - Ratio Verdaderos Positivos: Aquellos casos con daño, observando el *ground truth*, donde la
        predicción y *ground truth* es exactamente igual. Por ejemplo, p=[0,1,0], gt=[0,1,1] es incorrecto, mientras que
        p=[0,1,0], gt=[0,1,0] es correcto. Se suman los casos correctos se divide entre el número de casos 
        con al menos un daño.""")
    st.write(""" - Ratio Falsos Positivos: Aquellos casos sin daño, observando el *ground truth*, cuya predicción 
        contiene al menos un daño (un 1). Se suman estos casos y se divide entre el número de casos sin daño.""")
    tpr, fpr = [], []
    for threshold in np.arange(0, 1.01, 0.01):
        logits_thresholded = (logits > threshold)
        logits_thresholded = logits_thresholded.astype(np.uint8)

        # True positive rate: Aquellos casos con daño que son correctos (EXACTAMENTE igual predicción y ground truth)
        some_damage_indices = np.where((targets.sum(axis=1) > 0))[0]
        # suma_correctos ->  out = [[0,1,0], [0,1,1], [1,0,0]] // target = [[0,1,0], [1,1,0], [0,0,0]] // -> [3,1,2]
        suma_correctos = np.sum((logits_thresholded[some_damage_indices] == targets[some_damage_indices]) * 1, axis=1)
        # La suma de todos los correctos, para que sea estrictamente igual tiene que ser igual a 
        # la longitud que indica el número de clases, es decir, todas las clases en prediccion y target son iguales
        tpr.append(np.sum((suma_correctos == targets.shape[1]) * 1) / len(some_damage_indices))

        # False positive rate: Aquellos casos sin daño que en su predicción tienen algún tipo de daño/clase indicada
        no_damage_indices = np.where((targets.sum(axis=1) == 0))[0]
        # suma_incorrectos ->  [[0,1,0], [0,1,1], [0,0,0]] -> [1,2,0]
        suma_incorrectos = np.sum((logits_thresholded[no_damage_indices] > 0) * 1, axis=1)
        fpr.append(np.sum((suma_incorrectos > 0) * 1) / len(no_damage_indices))

    fig_roc2 = px.line(
        x=fpr, y=tpr, labels={
            "x": "Ratio Falsos Positivos",
            "y": "Ratio Verdaderos Positivos",
        },
    )

    st.plotly_chart(fig_roc2, use_container_width=True)


def redes():
    # Render the readme as markdown using st.markdown.
    st.markdown(get_file_content_as_string("info_redes.md"))
    st.markdown("## Ejemplo")
    st.sidebar.header('Panel de Control')

    df_truth = load_full_df("truth")
    df_predicted = load_full_df("predicted")

    # readme_text = st.markdown(get_file_content_as_string("instructions.md"))
    correct = st.sidebar.checkbox('Predicciones correctas', value=True)
    do_segmentation = st.sidebar.checkbox('Aplicar modelos Segmentación', value=False)
    struth, spredicted = filter_correct(df_truth, df_predicted, correct)

    selected_classes = st.sidebar.multiselect("¿Que clases deseas incluir en el análisis?", CLASSES, CLASSES)
    struth, spredicted = filter_classes(struth, spredicted, selected_classes)

    st.sidebar.markdown("# Ground Truth")
    truth_max = int(df_truth["sum"].max())
    truth_min = int(df_truth["sum"].min())
    min_t, max_t = st.sidebar.slider(
        "Cuántas clases (selecciona un rango)?", truth_min, truth_max, [truth_min, truth_max]
    )

    st.sidebar.markdown("# Predicciones")
    pred_max = int(df_predicted["sum"].max())
    pred_min = int(df_predicted["sum"].min())
    min_p, max_p = st.sidebar.slider(
        "Cuántas clases predichas (selecciona un rango)?", pred_min, pred_max, [pred_min, pred_max]
    )

    # Filtrar dependiendo los parámetros de los sliders anteriores.
    selected_frames = get_selected_frames(struth, spredicted, min_t, max_t, min_p, max_p)
    if len(selected_frames) < 1:
        st.error("No hay muestras que cumplan los requisitos especificados. Por favor especifica otros parámetros.")
        st.stop()

    if len(selected_frames) == 1:
        st.sidebar.write('Solo hay *una muestra* que cumpla los requisitos')
        selected_frame_index = 0
    else:
        selected_frame_index = st.sidebar.slider("Elige una muestra (indice)", 0, len(selected_frames) - 1, 0)

    # Display selected frame info and image
    selected_index = selected_frames[selected_frame_index]
    truth_info = struth.loc[selected_index]
    pred_info = spredicted.loc[selected_index]

    st.write(f"**Ground Truth** (`{truth_info['sum']}` clases)")
    truth_classes = get_item_classes(truth_info)
    st.write(" - ".join(truth_classes))
    st.write(f"**Predicho** (`{pred_info['sum']}` clases)")
    pred_classes = get_item_classes(pred_info)
    st.write(" - ".join(pred_classes))

    file_path = truth_info["file_path"]
    image = load_image(file_path)

    # --- SEGMENTATION CASE ---
    st.text("")  # Para añadir un poco de espacio con la parte de arriba
    # # Si queremos una única red para predecir las grietas descomentar lo siguiente
    # truth_classes = set([re.sub(r'(?<=Grietas).*$', "", d) for d in truth_classes])
    # pred_classes = set([re.sub(r'(?<=Grietas).*$', "", d) for d in pred_classes])
    parcheo_paths = load_segmentation_paths("Parcheo")
    hueco_paths = load_segmentation_paths("Hueco")
    longitudinales_paths = load_segmentation_paths("Grietas Longitudinales")
    transversales_paths = load_segmentation_paths("Grietas Transversales")

    parcheo_classes_color_map = {0: 0, 1: 1}
    hueco_classes_color_map = {0: 0, 1: 2}
    grietas_classes_color_map = {0: 0, 1: 3}
    overlay_alpha = 0.65
    overlay_cmap = 'prism'

    if do_segmentation and \
            (list(set(truth_classes) & set(SEGMENTATION_CLASSES)) or list(
                set(pred_classes) & set(SEGMENTATION_CLASSES))):

        num_plots, seg_classes = 1, []
        if "Parcheo" in truth_classes or "Parcheo" in pred_classes:
            seg_classes.append("Parcheo")
            num_plots += 1
        if "Hueco" in truth_classes or "Hueco" in pred_classes:
            seg_classes.append("Hueco")
            num_plots += 1
        if "Grietas transversales" in truth_classes or "Grietas transversales" in pred_classes:
            seg_classes.append("Grietas")
            num_plots += 1
        if ("Grietas longitudinales" in truth_classes or "Grietas longitudinales" in pred_classes) and \
                not "Grietas" in seg_classes:
            seg_classes.append("Grietas")
            num_plots += 1

        rows, cols = PLOTS2AX[num_plots]
        fig, ax = plt.subplots(rows, cols)
        plt.subplots_adjust(wspace=0.15, hspace=0.25)  # set the spacing between axes.
        positions = AX2POSITIONS[(rows, cols)]
        current_position = 0

        ax[positions[current_position]].imshow(image)
        ax[positions[current_position]].axis("off")
        ax[positions[current_position]].set_title("Imagen Original")
        current_position += 1

        original_h, original_w, _ = image.shape
        batch = batchify_image(image)

        # Segmentation inference
        if "Parcheo" in seg_classes:
            ax[positions[current_position]].imshow(image)

            parcheo_model = load_model_fold(parcheo_paths, file_path, 'Parcheo')
            parcheo_mask = predict_segmentation(parcheo_model, original_h, original_w, batch)
            parcheo_mask = map_mask_classes(parcheo_mask, parcheo_classes_color_map)
            parcheo_mask = np.ma.masked_where(parcheo_mask == 0, parcheo_mask)
            ax[positions[current_position]].imshow(
                parcheo_mask, overlay_cmap, interpolation='bilinear', alpha=overlay_alpha,
                vmin=0, vmax=len(SEGMENTATION_CLASSES)
            )

            ax[positions[current_position]].axis("off")
            ax[positions[current_position]].set_title("Parcheo")
            current_position += 1

        if "Hueco" in seg_classes:
            ax[positions[current_position]].imshow(image)

            hueco_model = load_model_fold(hueco_paths, file_path, 'Hueco')
            hueco_mask = predict_segmentation(hueco_model, original_h, original_w, batch)
            hueco_mask = map_mask_classes(hueco_mask, hueco_classes_color_map)
            hueco_mask = np.ma.masked_where(hueco_mask == 0, hueco_mask)
            ax[positions[current_position]].imshow(
                hueco_mask, overlay_cmap, interpolation='bilinear', alpha=overlay_alpha,
                vmin=0, vmax=len(SEGMENTATION_CLASSES)
            )

            ax[positions[current_position]].axis("off")
            ax[positions[current_position]].set_title("Hueco")
            current_position += 1

        if "Grietas" in seg_classes:

            ax[positions[current_position]].imshow(image)

            if "Grietas longitudinales" in truth_classes or "Grietas longitudinales" in pred_classes:
                longitudinal_model = load_model_fold(longitudinales_paths, file_path, 'Grietas Longitudinales')
                longitudinal_mask = predict_segmentation(longitudinal_model, original_h, original_w, batch)
                longitudinal_mask = map_mask_classes(longitudinal_mask, grietas_classes_color_map)
                longitudinal_mask = np.ma.masked_where(longitudinal_mask == 0, longitudinal_mask)
                ax[positions[current_position]].imshow(
                    longitudinal_mask, overlay_cmap, interpolation='bilinear', alpha=overlay_alpha,
                    vmin=0, vmax=len(SEGMENTATION_CLASSES)
                )
            if "Grietas transversales" in truth_classes or "Grietas transversales" in pred_classes:
                transversal_model = load_model_fold(transversales_paths, file_path, 'Grietas Transversales')
                transversal_mask = predict_segmentation(transversal_model, original_h, original_w, batch)
                transversal_mask = map_mask_classes(transversal_mask, grietas_classes_color_map)
                transversal_mask = np.ma.masked_where(transversal_mask == 0, transversal_mask)
                ax[positions[current_position]].imshow(
                    transversal_mask, overlay_cmap, interpolation='bilinear', alpha=overlay_alpha,
                    vmin=0, vmax=len(SEGMENTATION_CLASSES)
                )

            ax[positions[current_position]].axis("off")
            ax[positions[current_position]].set_title("Grietas")
            current_position += 1

        st.pyplot(fig)
    else:
        st.image(image.astype(np.uint8), use_column_width=True)


@st.cache
def load_model_fold(parcheo_paths, file_path, clase):
    file_fold = find_path_fold(parcheo_paths, file_path, clase, 'fold0')
    model_weights = torch.load(
        f"checkpoints/{clase}/{file_fold}/model_best_metric.pt",
        map_location=torch.device('cpu')
    )
    parcheo_model = ExtraSmallUNet(n_channels=3, n_classes=1)
    parcheo_model = torch.nn.DataParallel(parcheo_model, device_ids=range(torch.cuda.device_count()))
    parcheo_model.load_state_dict(model_weights)
    parcheo_model.eval()
    return parcheo_model


@st.cache
def load_full_df(prefix):
    """

    :param prefix: (str) prefijo para seleccionar los dataframes por nombre, que compondrán la salida
    :return: (pandas Dataframe) Dataframe compuesto por todos los dataframes de los diferentes folds de validación
    """
    res = []
    df_rootdir = "info"
    for subdir, dirs, files in os.walk(df_rootdir):
        for file in files:
            file_path = os.path.join(subdir, file)
            if prefix in file_path:
                res.append(pd.read_csv(file_path))
    res = pd.concat(res)
    res.reset_index(inplace=True, drop=True)
    return res


@st.cache
def load_segmentation_paths(clase):
    """

    :param clase: (str) Clase de la cuál cargar los fold validation data paths
    :return: (dict) diccionario del tipo {'fold0':[name_1, name_2...], 'fold1':[name_x, name_y...]}
    """
    prefix = f"validation_{clase}_data_paths"
    res = {}
    for subdir, dirs, files in os.walk(os.path.join("checkpoints", clase)):
        for file in files:
            file_path = os.path.join(subdir, file)
            if prefix in file_path:
                data_paths = torch.load(file_path)["data_paths"]
                # Deseamos quitar del path al extension: a/b/c/file_name.jpg ->  a/b/c/file_name
                data_paths = [os.path.splitext(d)[0] for d in data_paths]
                res[file_path[file_path.find("fold"):file_path.find("fold") + 5]] = data_paths
    return res


@st.cache
def find_path_fold(paths_dictionary, query_path, clase, default_value=None):
    """

    :param paths_dictionary: (dict) diccionario que contiene los paths pertenecientes a cada fold
    :param query_path: (str) path a buscar y determinar a que fold pertenece
    :param clase: (str) clase a la que pertenecen los paths
    :param default_value: (str) default fold a devolver si ninguno ha coincidido con le query
    :return: (str) 'foldN' a qué fold pertenece el query_path utilizando paths_dictionary
    """

    query_path = os.path.splitext(query_path)[0]
    query_path = query_path.replace("images", f"masks/{clase}")

    for key in paths_dictionary:
        if query_path in paths_dictionary[key]:
            return key
    if default_value is not None:
        return default_value
    return NoneFold


def batchify_image(image, img_size=512):
    transform = [
        albumentations.Resize(img_size, img_size),
    ]
    img_t, _ = apply_augmentations(copy.deepcopy(image), albumentations.Compose(transform), None, None)
    img_t = apply_normalization(img_t, "reescale")
    img_t = torch.from_numpy(img_t.transpose(2, 0, 1)).float()  # Transpose -> Channels first
    batch = torch.unsqueeze(img_t, 0)
    return batch


def predict_segmentation(model, original_h, original_w, image):
    binary_pred_mask = None
    with torch.no_grad():
        outputs = model(image)
    for indx, single_pred in enumerate(outputs):
        resize_transform = albumentations.Resize(original_h, original_w)
        pred_mask = resize_transform(image=torch.sigmoid(single_pred).squeeze(0).data.cpu().numpy())["image"]
        binary_pred_mask = np.where(pred_mask > 0.5, 1, 0).astype(np.int32)
        break

    return binary_pred_mask


@st.cache
def count_masks(directory="data"):
    """

    :param directory: (str) directorio donde buscar las mascaras
    :return: (dict) diccionario con clase como clave y valor el número encontrado de mascaras de dicha clase
    """
    res = {"Grietas longitudinales": 0, "Grietas transversales": 0, "Parcheo": 0, "Hueco": 0}
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            for key in res:
                if f"masks/{key}" in os.path.join(subdir, file):
                    res[key] += 1
    return res


@st.cache
def filter_correct(truth, predicted, is_correct):
    """

    :param truth: (pandas dataframe) Dataframe con los paths y clases ground truth (multi-etiqueta)
    :param predicted: (pandas dataframe) Dataframe con los paths y clases predichos (multi-etiqueta)
    :param is_correct: (bool) Indicador para filtrar por si las predicciones deben ser correctas (True) o no (False)
    :return: (list pandas dataframe) Datasets provistos filtrados por si las predicciones son correctas o no
    """
    return [
        truth[truth['correct'] == (is_correct * 1)],
        predicted[predicted['correct'] == (is_correct * 1)],
    ]


@st.cache
def filter_classes(truth, predicted, classes):
    """

    :param truth: (pandas dataframe) Dataframe con los paths y clases ground truth (multi-etiqueta)
    :param predicted: (pandas dataframe) Dataframe con los paths y clases predichos (multi-etiqueta)
    :param classes: (list) listado de aquellas clases que queremos tener en cuenta únicamente de 'truth' y 'predicted'
    :return: (list pandas dataframe) Datasets provistos filtrados por 'classes'
    """
    val, exp = 1, "|"

    # if len(classes) == len(CLASSES):  # Hemos seleccionado toas las clases, lo dejamos
    #     return [truth, predicted]
    if len(classes) == 0:  # No se ha seleccionado ninguna clase == sin daño -> todas las clases a 0!
        val = 0
        classes = CLASSES
        exp = "&"

    query = ""
    for clase in classes:
        if query:  # Si query ya tiene algún valor
            query += f" {exp} "
        if ' ' in clase:
            clase = f"`{clase}`"
        query += f"{clase}=={val}"

    return [
        truth.query(query),
        predicted.query(query)
    ]


# Select frames based on the selection in the sidebar
@st.cache(hash_funcs={np.ufunc: str})
def get_selected_frames(summary_truth, summary_pred, min_t, max_t, min_p, max_p):
    """

    :param summary_truth:
    :param summary_pred:
    :param min_t:
    :param max_t:
    :param min_p:
    :param max_p:
    :return:
    """
    # Filtramos los dataframes en función del numero de predicciones seleccionado en los sliders
    truth_index = summary_truth[np.logical_and(summary_truth['sum'] >= min_t, summary_truth['sum'] <= max_t)].index
    preds_index = summary_pred[np.logical_and(summary_pred['sum'] >= min_p, summary_pred['sum'] <= max_p)].index
    # Solo tomamos los frames coincidentes en truths y preds
    return list(set(truth_index.tolist()).intersection(preds_index.tolist()))


# We can reuse the images across runs.
@st.cache(show_spinner=False)
def load_image(path):
    """

    :param path: (str) Path a imagen a cargar
    :return: (numpy array) Imagen cargada a través del 'path' como numpy array
    """
    return io.imread(path)


@st.cache
def get_item_classes(item):
    """

    :param item: (pandas dataframe row) fila ded un dataframe (caso multi-etiqueta)
    :return: (list) lista con todas las clases (1s) que contiene la fila
    """
    res = []
    for clase, val in item.iteritems():
        if val == 1 and clase in CLASSES:
            res.append(clase)
    if not len(res):
        return ["Sin daño"]
    return res


@st.cache
def load_multilabel_predictions():
    return np.load("todos_targets.npy"), np.load("todos_logits.npy")


if __name__ == "__main__":
    main()
