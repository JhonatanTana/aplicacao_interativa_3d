from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

import numpy as np
import math
import sys

# ============================================================
# CONFIGURAÇÕES
# ============================================================

WIDTH = 1000
HEIGHT = 700

# Tamanho solicitado
BASE_SIZE = 1.0
HEIGHT_3D = 1.5

# Cor azul-bebê
BABY_BLUE = (
    0.63,
    0.78,
    0.92,
    1.0
)


# ============================================================
# QUATERNION
# ============================================================

# [w, x, y, z]
quaternion = np.array(
    [1.0, 0.0, 0.0, 0.0],
    dtype=np.float64
)


# ============================================================
# MOUSE
# ============================================================

mouse_down = False

last_x = 0
last_y = 0


# ============================================================
# CONFIGURAÇÃO DA ILUMINAÇÃO
#
# ESPECIFICAÇÕES EXATAS
# ============================================================

def configurar_iluminacao():

    # Iluminação
    glEnable(GL_LIGHTING)

    # Luz 0
    glEnable(GL_LIGHT0)

    # Profundidade
    glEnable(GL_DEPTH_TEST)

    # Material acompanha glColor
    glEnable(GL_COLOR_MATERIAL)

    glColorMaterial(
        GL_FRONT_AND_BACK,
        GL_AMBIENT_AND_DIFFUSE
    )

    # --------------------------------------------------------
    # AMBIENTE
    # --------------------------------------------------------

    glLightfv(
        GL_LIGHT0,
        GL_AMBIENT,
        [0.2, 0.2, 0.2, 1.0]
    )

    # --------------------------------------------------------
    # DIFUSA
    # --------------------------------------------------------

    glLightfv(
        GL_LIGHT0,
        GL_DIFFUSE,
        [0.8, 0.8, 0.8, 1.0]
    )

    # --------------------------------------------------------
    # ESPECULAR
    # --------------------------------------------------------

    glLightfv(
        GL_LIGHT0,
        GL_SPECULAR,
        [1.0, 1.0, 1.0, 1.0]
    )

    # --------------------------------------------------------
    # POSIÇÃO DA LUZ
    # --------------------------------------------------------

    glLightfv(
        GL_LIGHT0,
        GL_POSITION,
        [5.0, 10.0, 5.0, 1.0]
    )

    # --------------------------------------------------------
    # MATERIAL ESPECULAR
    # --------------------------------------------------------

    glMaterialfv(
        GL_FRONT_AND_BACK,
        GL_SPECULAR,
        [1.0, 1.0, 1.0, 1.0]
    )

    # --------------------------------------------------------
    # BRILHO
    # --------------------------------------------------------

    glMaterialf(
        GL_FRONT_AND_BACK,
        GL_SHININESS,
        50.0
    )


# ============================================================
# MULTIPLICAÇÃO DE QUATERNIONS
# ============================================================

def quaternion_multiply(q1, q2):

    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array([

        w1*w2
        - x1*x2
        - y1*y2
        - z1*z2,

        w1*x2
        + x1*w2
        + y1*z2
        - z1*y2,

        w1*y2
        - x1*z2
        + y1*w2
        + z1*x2,

        w1*z2
        + x1*y2
        - y1*x2
        + z1*w2

    ], dtype=np.float64)


# ============================================================
# NORMALIZAR QUATERNION
# ============================================================

def normalize_quaternion(q):

    norm = np.linalg.norm(q)

    if norm == 0:

        return np.array(
            [1.0, 0.0, 0.0, 0.0],
            dtype=np.float64
        )

    return q / norm


# ============================================================
# QUATERNION -> MATRIZ 4x4
# ============================================================

def quaternion_to_matrix(q):

    q = normalize_quaternion(q)

    w, x, y, z = q

    return np.array([

        [
            1 - 2*(y*y + z*z),
            2*(x*y - z*w),
            2*(x*z + y*w),
            0
        ],

        [
            2*(x*y + z*w),
            1 - 2*(x*x + z*z),
            2*(y*z - x*w),
            0
        ],

        [
            2*(x*z - y*w),
            2*(y*z + x*w),
            1 - 2*(x*x + y*y),
            0
        ],

        [
            0,
            0,
            0,
            1
        ]

    ], dtype=np.float32)


# ============================================================
# QUATERNION A PARTIR DE EIXO + ÂNGULO
# ============================================================

def quaternion_from_axis_angle(axis, angle):

    axis = np.array(
        axis,
        dtype=np.float64
    )

    axis /= np.linalg.norm(axis)

    half = angle / 2.0

    s = math.sin(half)

    return np.array([

        math.cos(half),
        axis[0] * s,
        axis[1] * s,
        axis[2] * s

    ], dtype=np.float64)


# ============================================================
# DESENHAR PIRÂMIDE
# ============================================================

def desenhar_piramide():

    # ========================================================
    # MATRIZ DA PIRÂMIDE
    # ========================================================

    glPushMatrix()

    # Aplicar quaternion
    rotation_matrix = quaternion_to_matrix(
        quaternion
    )

    glMultMatrixf(
        rotation_matrix.T
    )

    # ========================================================
    # COR AZUL-BEBÊ
    # ========================================================

    glColor4f(
        BABY_BLUE[0],
        BABY_BLUE[1],
        BABY_BLUE[2],
        BABY_BLUE[3]
    )

    # ========================================================
    # PIRÂMIDE
    # ========================================================

    glBegin(GL_TRIANGLES)

    # --------------------------------------------------------
    # FACE FRONTAL
    # --------------------------------------------------------

    glNormal3f(
        0.0,
        0.555,
        -0.832
    )

    glVertex3f(
        -BASE_SIZE,
        -HEIGHT_3D / 2,
        -BASE_SIZE
    )

    glVertex3f(
        BASE_SIZE,
        -HEIGHT_3D / 2,
        -BASE_SIZE
    )

    glVertex3f(
        0.0,
        HEIGHT_3D / 2,
        0.0
    )

    # --------------------------------------------------------
    # FACE DIREITA
    # --------------------------------------------------------

    glNormal3f(
        0.832,
        0.555,
        0.0
    )

    glVertex3f(
        BASE_SIZE,
        -HEIGHT_3D / 2,
        -BASE_SIZE
    )

    glVertex3f(
        BASE_SIZE,
        -HEIGHT_3D / 2,
        BASE_SIZE
    )

    glVertex3f(
        0.0,
        HEIGHT_3D / 2,
        0.0
    )

    # --------------------------------------------------------
    # FACE TRASEIRA
    # --------------------------------------------------------

    glNormal3f(
        0.0,
        0.555,
        0.832
    )

    glVertex3f(
        BASE_SIZE,
        -HEIGHT_3D / 2,
        BASE_SIZE
    )

    glVertex3f(
        -BASE_SIZE,
        -HEIGHT_3D / 2,
        BASE_SIZE
    )

    glVertex3f(
        0.0,
        HEIGHT_3D / 2,
        0.0
    )

    # --------------------------------------------------------
    # FACE ESQUERDA
    # --------------------------------------------------------

    glNormal3f(
        -0.832,
        0.555,
        0.0
    )

    glVertex3f(
        -BASE_SIZE,
        -HEIGHT_3D / 2,
        BASE_SIZE
    )

    glVertex3f(
        -BASE_SIZE,
        -HEIGHT_3D / 2,
        -BASE_SIZE
    )

    glVertex3f(
        0.0,
        HEIGHT_3D / 2,
        0.0
    )

    glEnd()

    # ========================================================
    # BASE QUADRADA
    # ========================================================

    glBegin(GL_QUADS)

    glNormal3f(
        0.0,
        -1.0,
        0.0
    )

    glVertex3f(
        -BASE_SIZE,
        -HEIGHT_3D / 2,
        -BASE_SIZE
    )

    glVertex3f(
        -BASE_SIZE,
        -HEIGHT_3D / 2,
        BASE_SIZE
    )

    glVertex3f(
        BASE_SIZE,
        -HEIGHT_3D / 2,
        BASE_SIZE
    )

    glVertex3f(
        BASE_SIZE,
        -HEIGHT_3D / 2,
        -BASE_SIZE
    )

    glEnd()

    glPopMatrix()


# ============================================================
# CONFIGURAÇÃO DA CÂMERA
# ============================================================

def configurar_camera():

    glMatrixMode(GL_PROJECTION)

    glLoadIdentity()

    gluPerspective(
        45.0,
        WIDTH / HEIGHT,
        0.1,
        100.0
    )

    glMatrixMode(GL_MODELVIEW)

    glLoadIdentity()

    # Câmera olhando para o centro
    gluLookAt(

        0.0,
        0.0,
        8.0,

        0.0,
        0.0,
        0.0,

        0.0,
        1.0,
        0.0
    )


# ============================================================
# DISPLAY
# ============================================================

def display():

    glClear(
        GL_COLOR_BUFFER_BIT |
        GL_DEPTH_BUFFER_BIT
    )

    glLoadIdentity()

    # Câmera
    gluLookAt(

        0.0,
        0.0,
        8.0,

        0.0,
        0.0,
        0.0,

        0.0,
        1.0,
        0.0
    )

    # Luz permanece fixa no mundo
    glLightfv(
        GL_LIGHT0,
        GL_POSITION,
        [5.0, 5.0, 5.0, 1.0]
    )

    # Pirâmide
    desenhar_piramide()

    # ========================================================
    # LEGENDAS
    # ========================================================

    glDisable(GL_LIGHTING)

    glMatrixMode(GL_PROJECTION)

    glPushMatrix()

    glLoadIdentity()

    glOrtho(
        0,
        WIDTH,
        0,
        HEIGHT,
        -1,
        1
    )

    glMatrixMode(GL_MODELVIEW)

    glPushMatrix()

    glLoadIdentity()

    # Texto 1
    desenhar_texto(
        25,
        HEIGHT - 40,
        "Clique e arraste para girar"
    )

    # Texto 2
    desenhar_texto(
        25,
        HEIGHT - 75,
        "R = Resetar | ESC = Sair"
    )

    glPopMatrix()

    glMatrixMode(GL_PROJECTION)

    glPopMatrix()

    glMatrixMode(GL_MODELVIEW)

    glEnable(GL_LIGHTING)

    glutSwapBuffers()


# ============================================================
# TEXTO
# ============================================================

def desenhar_texto(
    x,
    y,
    texto
):

    glColor3f(
        1.0,
        1.0,
        1.0
    )

    glRasterPos2f(
        x,
        y
    )

    for char in texto:

        glutBitmapCharacter(
            GLUT_BITMAP_HELVETICA_18,
            ord(char)
        )


# ============================================================
# RESIZE
# ============================================================

def reshape(
    width,
    height
):

    global WIDTH
    global HEIGHT

    WIDTH = max(
        width,
        1
    )

    HEIGHT = max(
        height,
        1
    )

    glViewport(
        0,
        0,
        WIDTH,
        HEIGHT
    )

    configurar_camera()


# ============================================================
# MOUSE
# ============================================================

def mouse(
    button,
    state,
    x,
    y
):

    global mouse_down
    global last_x
    global last_y

    if button == GLUT_LEFT_BUTTON:

        if state == GLUT_DOWN:

            mouse_down = True

            last_x = x
            last_y = y

        elif state == GLUT_UP:

            mouse_down = False


# ============================================================
# MOVIMENTO DO MOUSE
# ============================================================

def motion(
    x,
    y
):

    global last_x
    global last_y
    global quaternion

    if not mouse_down:
        return

    dx = x - last_x
    dy = y - last_y

    sensitivity = 0.01

    # --------------------------------------------------------
    # Rotação horizontal
    # --------------------------------------------------------

    q_y = quaternion_from_axis_angle(
        [0.0, 1.0, 0.0],
        dx * sensitivity
    )

    # --------------------------------------------------------
    # Rotação vertical
    # --------------------------------------------------------

    q_x = quaternion_from_axis_angle(
        [1.0, 0.0, 0.0],
        dy * sensitivity
    )

    # --------------------------------------------------------
    # Aplicar rotações
    # --------------------------------------------------------

    quaternion = quaternion_multiply(
        q_y,
        quaternion
    )

    quaternion = quaternion_multiply(
        quaternion,
        q_x
    )

    quaternion = normalize_quaternion(
        quaternion
    )

    last_x = x
    last_y = y

    glutPostRedisplay()


# ============================================================
# TECLADO
# ============================================================

def keyboard(
    key,
    x,
    y
):

    global quaternion

    # ESC
    if key == b'\x1b':

        sys.exit()

    # R
    if key in (b'r', b'R'):

        quaternion = np.array(
            [1.0, 0.0, 0.0, 0.0],
            dtype=np.float64
        )

        glutPostRedisplay()


# ============================================================
# INICIALIZAÇÃO
# ============================================================

def main():

    glutInit(
        sys.argv
    )

    glutInitDisplayMode(
        GLUT_DOUBLE |
        GLUT_RGB |
        GLUT_DEPTH
    )

    glutInitWindowSize(
        WIDTH,
        HEIGHT
    )

    glutInitWindowPosition(
        100,
        50
    )

    glutCreateWindow(
        b"Piramide 3D - PyOpenGL"
    )

    # Fundo
    glClearColor(
        0.10,
        0.10,
        0.10,
        1.0
    )

    # Configurações
    configurar_iluminacao()

    configurar_camera()

    # ========================================================
    # CALLBACKS
    # ========================================================

    glutDisplayFunc(
        display
    )

    glutReshapeFunc(
        reshape
    )

    glutMouseFunc(
        mouse
    )

    glutMotionFunc(
        motion
    )

    glutKeyboardFunc(
        keyboard
    )

    # ========================================================
    # LOOP
    # ========================================================

    glutMainLoop()


# ============================================================
# EXECUTAR
# ============================================================

if __name__ == "__main__":

    main()