"""Project Lab border points onto the exact triangles exported by the renderer."""
from __future__ import annotations

import math
import struct
from array import array
from collections import defaultdict
from pathlib import Path


class MeshLayer:
    def __init__(self, vertices: list[tuple[float, ...]], indices: tuple[int, ...],
                 column: int, row: int):
        self.vertices = vertices
        self.triangles = [indices[i:i + 3] for i in range(0, len(indices), 3)]
        self.bins: dict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)
        for triangle in self.triangles:
            points = [vertices[index] for index in triangle]
            x0, x1 = min(p[0] for p in points), max(p[0] for p in points)
            y0, y1 = min(p[1] for p in points), max(p[1] for p in points)
            for bx in range(max(0, math.floor((x0-column)*32-0.001)),
                            min(31, math.floor((x1-column)*32+0.001))+1):
                for by in range(max(0, math.floor((y0-row)*32-0.001)),
                                min(31, math.floor((y1-row)*32+0.001))+1):
                    self.bins[bx, by].append(triangle)

    def sample(self, world_x: float, world_y: float, column: int, row: int):
        bx = max(0, min(31, math.floor((world_x-column)*32)))
        by = max(0, min(31, math.floor((world_y-row)*32)))
        for triangle in self.bins.get((bx, by), ()):
            a, b, c = (self.vertices[index] for index in triangle)
            denominator = (b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
            if abs(denominator) < 1e-10:
                continue
            wa = ((b[1]-c[1])*(world_x-c[0])+(c[0]-b[0])*(world_y-c[1]))/denominator
            wb = ((c[1]-a[1])*(world_x-c[0])+(a[0]-c[0])*(world_y-c[1]))/denominator
            wc = 1-wa-wb
            if min(wa, wb, wc) < -0.00005:
                continue
            return wa*a[2]+wb*b[2]+wc*c[2]
        return None


class GroundSurface:
    def __init__(self, prefix: Path, tile_width: int):
        self.prefix = prefix
        self.tiles = {}
        paths = sorted(path for path in prefix.parent.glob(prefix.name + ".*_*.bin")
                       if not path.name.startswith(prefix.name + ".depth."))
        if not paths:
            raise ValueError("renderer ground mesh export is missing")
        for path in paths:
            data = path.read_bytes()
            if data[:8] != b"C3XBRD1\0":
                raise ValueError("unrecognized renderer ground mesh export")
            tile_x, tile_y = struct.unpack_from("<2i", data, 8)
            if (tile_x+tile_y) % 2:
                raise ValueError("invalid renderer ground mesh tile")
            offset = 16
            column, row = (tile_x+tile_y)//2, (tile_x-tile_y)//2
            layers = []
            for _ in range(2):
                vertex_count, index_count = struct.unpack_from("<2I", data, offset)
                offset += 8
                if vertex_count > 50000 or index_count > 300000 or index_count % 3:
                    raise ValueError("invalid renderer ground mesh counts")
                vertices = list(struct.iter_unpack("<3f", data[offset:offset+vertex_count*12]))
                offset += vertex_count*12
                indices = struct.unpack_from(f"<{index_count}I", data, offset)
                offset += index_count*4
                if any(index >= vertex_count for index in indices):
                    raise ValueError("renderer ground mesh index is out of range")
                layers.append(MeshLayer(vertices, indices, column, row))
            if offset != len(data):
                raise ValueError("renderer ground mesh export has trailing bytes")
            self.tiles[column, row] = layers

    def sample_world(self, world_x: float, world_y: float) -> float:
        columns = {math.floor(world_x-0.00001), math.floor(world_x+0.00001)}
        rows = {math.floor(world_y-0.00001), math.floor(world_y+0.00001)}
        found = []
        for column in columns:
            for row in rows:
                layers = self.tiles.get((column, row))
                if layers is None:
                    continue
                # Layer 2 replaces regular ground through mountain ranges.
                for layer in reversed(layers):
                    sampled = layer.sample(world_x, world_y, column, row)
                    if sampled is not None:
                        found.append(sampled)
                        break
        if not found:
            raise ValueError(f"border point misses exported ground mesh at {world_x:.5f},{world_y:.5f}")
        return max(found)

    def project(self, screen_x: float, screen_y: float, image_size: tuple[int, int],
                tile_width: int, center: tuple[int, int]) -> tuple[float, float]:
        projected_x, projected_y, _ = self.project_with_depth(
            screen_x, screen_y, image_size, tile_width, center)
        return projected_x, projected_y

    def project_with_depth(self, screen_x: float, screen_y: float,
                           image_size: tuple[int, int], tile_width: int,
                           center: tuple[int, int]) -> tuple[float, float, float]:
        dx = (screen_x-image_size[0]/2)/(tile_width/2)
        dy = (screen_y-image_size[1]/2)/(tile_width/4)
        column = (center[0]+center[1])/2+0.5+(dx+dy)/2
        row = (center[0]-center[1])/2+0.5+(dx-dy)/2
        height = self.sample_world(column, row)*112-2.5
        return (screen_x, screen_y-height*(tile_width/224*0.82),
                screen_y+height*0.0016*image_size[1])


class ProjectedTerrain:
    """A screen-space depth lookup of the exported production ground triangles."""
    BIN_SIZE = 12

    def __init__(self, surface: GroundSurface, image_size: tuple[int, int],
                 tile_width: int, center: tuple[int, int]):
        self.bins = defaultdict(list)
        self.image_size = image_size
        center_column = (center[0]+center[1])/2+0.5
        center_row = (center[0]-center[1])/2+0.5
        for layers in surface.tiles.values():
            for layer in layers:
                projected = []
                for world_x, world_y, world_z in layer.vertices:
                    flat_x = image_size[0]/2+(world_x+world_y-center_column-center_row)*tile_width/2
                    flat_y = image_size[1]/2+(world_x-world_y-center_column+center_row)*tile_width/4
                    height = world_z*112-2.5
                    projected.append((flat_x, flat_y-height*(tile_width/224*0.82),
                                      flat_y+height*0.0016*image_size[1]))
                for indices in layer.triangles:
                    triangle = tuple(projected[index] for index in indices)
                    x0, x1 = min(p[0] for p in triangle), max(p[0] for p in triangle)
                    y0, y1 = min(p[1] for p in triangle), max(p[1] for p in triangle)
                    if x1 < 0 or y1 < 0 or x0 >= image_size[0] or y0 >= image_size[1]:
                        continue
                    for bx in range(max(0, math.floor(x0/self.BIN_SIZE)),
                                    min((image_size[0]-1)//self.BIN_SIZE,
                                        math.floor(x1/self.BIN_SIZE))+1):
                        for by in range(max(0, math.floor(y0/self.BIN_SIZE)),
                                        min((image_size[1]-1)//self.BIN_SIZE,
                                            math.floor(y1/self.BIN_SIZE))+1):
                            self.bins[bx, by].append(triangle)

    def front_depth(self, screen_x: float, screen_y: float) -> float | None:
        front = None
        for triangle in self.bins.get((math.floor(screen_x/self.BIN_SIZE),
                                       math.floor(screen_y/self.BIN_SIZE)), ()):
            a, b, c = triangle
            denominator = (b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
            if abs(denominator) < 1e-10:
                continue
            wa = ((b[1]-c[1])*(screen_x-c[0])+(c[0]-b[0])*(screen_y-c[1]))/denominator
            wb = ((c[1]-a[1])*(screen_x-c[0])+(a[0]-c[0])*(screen_y-c[1]))/denominator
            wc = 1-wa-wb
            if min(wa, wb, wc) < -0.00001:
                continue
            depth = wa*a[2]+wb*b[2]+wc*c[2]
            if front is None or depth > front:
                front = depth
        return front

    def is_occluded(self, screen_x: float, screen_y: float,
                    line_depth: float, clearance: float = 5.0) -> bool:
        front = self.front_depth(screen_x, screen_y)
        return front is not None and front > line_depth+clearance


class RenderDepth:
    """Depth of the finished production city pass, including foreground art."""
    def __init__(self, prefix: Path, image_size: tuple[int, int]):
        paths = sorted(prefix.parent.glob(prefix.name + ".depth.*_*.bin"))
        if not paths:
            raise ValueError("renderer depth export is missing")
        self.width, self.height = image_size
        self.samples = array("f", [1.0])*int(self.width*self.height)
        self.offset = None
        for path in paths:
            data = path.read_bytes()
            if data[:8] != b"C3XBDP1\0":
                raise ValueError("unrecognized renderer depth export")
            left, top, width, height = struct.unpack_from("<4i", data, 8)
            offset, = struct.unpack_from("<f", data, 24)
            if (left < 0 or top < 0 or width < 1 or height < 1 or
                left+width > self.width or top+height > self.height or
                len(data) != 28+width*height*4):
                raise ValueError("invalid renderer depth rectangle")
            if self.offset is not None and self.offset != offset:
                raise ValueError("inconsistent renderer depth origin")
            self.offset = offset
            for y in range(height):
                row = struct.unpack_from(f"<{width}f", data, 28+y*width*4)
                start = (top+y)*self.width+left
                self.samples[start:start+width] = array("f", row)

    def is_occluded(self, screen_x: float, screen_y: float,
                    line_depth: float, clearance: float = 5.0) -> bool:
        x, y = math.floor(screen_x), math.floor(screen_y)
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False
        line_z = 0.5-(line_depth-self.offset)/16384
        return self.samples[y*self.width+x] < line_z-clearance/16384
