import streamlit as st
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import os
import time
from PIL import Image
import SessionState
from streamlit.report_thread import get_report_ctx


class TransformationDefinition:
    def __init__(self, name, params=[]):
        self.name = name
        self.params = params
        self.documentation = None
        documentation_file = './templates/{}.md'.format(name)
        if os.path.exists(documentation_file):
            print("found {}".format(documentation_file))
            with open('./templates/{}.md'.format(name)) as f:
                self.documentation = f.read()

class NumberRange:
    def __init__(self, name, min=0., max=1.):
        self.name = name
        self.min = min
        self.max = max

    def render(self, handle, key=None):
        return handle.slider(self.name, min_value=self.min, max_value=self.max, key=key)

class NumberInput:
    def __init__(self, name, min=None, max=None):
        self.name = name
        self.min = min
        self.max = max

    def render(self, handle, key=None):
        return handle.number_input(self.name, min_value=self.min, max_value=self.max, key=key)

class TupleInput:
    def __init__(self, name, input_1, input_2):
        self.name = name
        self.input_1 = input_1
        self.input_2 = input_2

    def render(self, handle, key=None):
        output_1 = self.input_1.render(handle, key=key)
        output_2 = self.input_2.render(handle, key=key)
        return (output_1, output_2)

class CategoricalInput:
    def __init__(self, name, categories, category_names=None):
        self.name = name
        self.categories = categories

    def render(self, handle, key=None):
        output = handle.selectbox(self.name, self.categories, key=key)
        return output

class BooleanInput:
    def __init__(self, name):
        self.name = name

    def render(self, handle, key=None):
        return handle.checkbox(self.name, key=key)

class Transformation:
    def __init__(self, name, params={}):
        self.name = name
        self.params = params

    def __str__(self):
        param_string = ", ".join(["{}={}".format(name, value) for name, value in self.params.items()])
        return "transforms.{}({})".format(self.name, param_string)

class Transformations:
    def __init__(self, transformations):
        self.transformations = transformations

    def to_pytorch(self):
        pytorch_transformations = []
        for transformation in self.transformations:
            transform_class = getattr(transforms, transformation.name)
            pytorch_transformations.append(transform_class(**transformation.params))

        return transforms.Compose(pytorch_transformations)

    def __str__(self):
        return """
from torchvision import transforms
from torchvision.transforms import InterpolationMode

transform = transforms.Compose([
{}
])
        """.format(",\n".join(["\t{}".format(str(t)) for t in state.applied_transforms]))

TRANSFORMS_SUPPORTED = [
    TransformationDefinition("Grayscale", params=[
        CategoricalInput('num_output_channels', [1, 3])
    ]),
    TransformationDefinition("ColorJitter", params=[NumberRange("brightness"), NumberRange("contrast"), NumberRange("saturation"), NumberRange("hue", min=-0., max=0.5)]),
    TransformationDefinition("CenterCrop", params=[NumberInput("size", min=0, max=None)]),
    TransformationDefinition("Pad", params=[
        NumberInput("padding", min=0, max=None),
        NumberInput("fill", min=0, max=None),
        CategoricalInput('padding_mode', ['constant', 'edge', 'reflect', 'symmetric']),
    ]),
    TransformationDefinition("RandomAffine", params=[
        NumberRange("degrees", min=0, max=360),
        TupleInput("translate", NumberRange("translate_1"), NumberRange("translate_2")),
        TupleInput("scale", NumberRange("scale_1", max=10.), NumberRange("scale_2", max=10.)),
        TupleInput("shear", NumberRange("shear_1"), NumberRange("shear_2")),
        NumberInput("fill", min=0, max=None),
        CategoricalInput('interpolation', [InterpolationMode.NEAREST, InterpolationMode.BILINEAR, InterpolationMode.BICUBIC]),
    ]),
    TransformationDefinition("RandomCrop", params=[
        NumberInput("size", min=0),
        BooleanInput("pad_if_needed"),
        CategoricalInput('padding_mode', ['constant', 'edge', 'reflect', 'symmetric']),
    ]),

    TransformationDefinition("RandomGrayscale", params=[NumberRange("p")]),

    TransformationDefinition("RandomHorizontalFlip", params=[NumberRange("p")]),
    TransformationDefinition("RandomPerspective", params=[
        NumberRange("distortion_scale"),
        NumberRange("p"),
        NumberInput("fill", min=0, max=None),
        CategoricalInput('interpolation', [InterpolationMode.NEAREST, InterpolationMode.BILINEAR, InterpolationMode.BICUBIC]),
    ]),
    TransformationDefinition("RandomResizedCrop", params=[
        NumberInput("size", min=0, max=None),
        TupleInput("scale", NumberRange("scale_1", max=2.), NumberRange("scale_2", max=2.)),
        TupleInput("ratio", NumberRange("ratio_1", max=2.), NumberRange("ratio_2", max=2.)),
        CategoricalInput('interpolation', [InterpolationMode.NEAREST, InterpolationMode.BILINEAR, InterpolationMode.BICUBIC]),
    ]),
    TransformationDefinition("RandomRotation", params=[
        NumberRange("degrees", min=0, max=360),
        BooleanInput("expand"),
        TupleInput("center", NumberInput("center_1", min=0), NumberInput("center_1", min=0)),
        CategoricalInput('interpolation', [InterpolationMode.NEAREST, InterpolationMode.BILINEAR, InterpolationMode.BICUBIC]),
        NumberInput("fill", min=0, max=None),
    ]),
    TransformationDefinition("RandomVerticalFlip", params=[NumberRange("p")]),
    TransformationDefinition("Resize", params=[
        NumberInput("size", min=0, max=None),
        CategoricalInput('interpolation', [InterpolationMode.NEAREST, InterpolationMode.BILINEAR, InterpolationMode.BICUBIC]),
    ]),
    TransformationDefinition("GaussianBlur", params=[
        NumberInput("kernel_size", min=0, max=100),
        TupleInput("sigma", NumberRange("sigma_1", min=0., max=50.0), NumberRange("sigma_2", min=0., max=50.0)),
    ]),
    # Transforms on torch.*Tensor only
    TransformationDefinition("ToTensor", params=[]),
    TransformationDefinition("ToPILImage", params=[]),
    TransformationDefinition("RandomErasing", params=[
        NumberRange("p"),
        TupleInput("scale", NumberRange("scale_1", max=1.), NumberRange("scale_2", max=1.)),
        TupleInput("ratio", NumberRange("ratio_1", max=2.), NumberRange("ratio_2", max=2.)),
        BooleanInput("inplace"),
    ]),
]
# Clean up
image_path = None
state = SessionState.get(applied_transforms=[])

# Selection input
transform_names = [""] + [t.name for t in TRANSFORMS_SUPPORTED]
selection = st.sidebar.selectbox("Select Transformation", transform_names)
if selection != "":
    # Render options
    transform = [t for t in TRANSFORMS_SUPPORTED if t.name == selection][0]
    if transform.documentation:
        st.sidebar.markdown(transform.documentation)

    param_settings = {}
    # Render transform options
    for param_type in transform.params:
        # Bit of a hack here
        same_components = [t for t in state.applied_transforms if t.name == selection]
        if len(same_components) <= 1:
            key_idx = 0
        else:
            key_idx = len(same_components)
        param_settings[param_type.name] = param_type.render(st.sidebar, key="{}-{}-{}".format(selection, param_type.name, key_idx))

    if st.sidebar.button("Add {}".format(selection)):
        state.applied_transforms.append(Transformation(selection, param_settings))

if len(state.applied_transforms) > 0:
    if selection == state.applied_transforms[-1].name:
        if st.sidebar.button("Update {}".format(state.applied_transforms[-1].name)):
            state.applied_transforms = state.applied_transforms[:-1]
            state.applied_transforms.append(Transformation(selection, param_settings))

    if st.sidebar.button("Remove {}".format(state.applied_transforms[-1].name)):
        state.applied_transforms = state.applied_transforms[:-1]

    if st.sidebar.button("Remove All"):
        state.applied_transforms = []
else:
    with open('./templates/instructions.md', 'r') as f:
        st.markdown(f.read())

transformations = Transformations(state.applied_transforms)
st.code(str(transformations))

# Get Image
st.sidebar.markdown("## 2. Select an Image")
selected_image_path = st.sidebar.selectbox("Choose an Image", sorted(os.listdir('./examples')))

st.sidebar.markdown("*OR*")

uploaded_image = st.sidebar.file_uploader("Upload Image")

ctx = get_report_ctx()
saved_image = None

if uploaded_image is not None:
    if uploaded_image.type != 'image/png':
        st.text("File type {} not supported".format(uploaded_image.type))
        st.stop()

    source_image = Image.open(uploaded_image)
    saved_image = os.path.join('./data/uploads', "{}-currentimg.png".format(ctx.session_id))
elif selected_image_path != '':
    source_image = Image.open(os.path.join('./examples/', selected_image_path))
    saved_image = os.path.join('./data/uploads', "{}-currentimg.png".format(ctx.session_id))

if saved_image:
    transform = transformations.to_pytorch()
    transformed_image = transform(source_image)
    width, height = transformed_image.size
    transformed_image = transformed_image.resize((500, int(height * 500 / width)))
    # TODO check if there are transformed_images
    transformed_image.save(saved_image)

    if len(state.applied_transforms) > 0:
        if st.button("Regenerate"):
            # This will re-run the transformation
            pass

    st.image(saved_image)
