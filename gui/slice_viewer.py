# -*- coding: utf-8 -*-
"""
Module: slice_viewer.py
Author: Gemini
Description: A PyQt6 widget for displaying and interacting with slice images.
             Supports panning and zooming.
"""

import numpy as np
from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt

class SliceViewer(QGraphicsView):
    """
    An interactive image viewer widget that displays a NumPy array.
    """
    def __init__(self):
        super().__init__()
        
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        
        # This item will hold our image
        self._pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self._pixmap_item)
        
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        # Enable panning with the left mouse button
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        
        self.zoom_factor = 1.25

    def set_image(self, image_array: np.ndarray | None):
        """
        Displays a NumPy array as an image.

        Args:
            image_array (np.ndarray | None): The grayscale image data to display.
                                             If None, the view is cleared.
        """
        if image_array is None:
            self._pixmap_item.setPixmap(QPixmap())
            return

        if image_array.ndim != 2 or image_array.dtype != np.uint8:
            raise TypeError("Image must be a 2D NumPy array of type uint8.")

        height, width = image_array.shape
        bytes_per_line = width
        
        # Create a QImage from the NumPy array
        q_image = QImage(
            image_array.data, 
            width, 
            height, 
            bytes_per_line, 
            QImage.Format.Format_Grayscale8
        )
        
        pixmap = QPixmap.fromImage(q_image)
        self._pixmap_item.setPixmap(pixmap)
        
        # Fit the image to the view on initial display
        self.fitInView(self.scene.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event):
        """Overrides the mouse wheel event to handle zooming."""
        if event.angleDelta().y() > 0:
            # Zoom in
            self.scale(self.zoom_factor, self.zoom_factor)
        else:
            # Zoom out
            self.scale(1 / self.zoom_factor, 1 / self.zoom_factor)

