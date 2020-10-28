import sys
from PIL import Image

def concat(G):
    images = [Image.open('raster/frame{:05d}.png'.format(x)) for x in range(1, (G * G)/2 + 1)] # [1, 801)
    #images = [Image.open('raster/frame{:05d}.png'.format(x)) for x in range((G * G) / 2 + 1, G * G + 1)] #[801, 1601)
    print len(images)

    total_width = G * 100
    total_height = G * 50

    new_im = Image.new('RGB', (total_width, total_height))

    x_offset = 0
    y_offset = 0
    for i, im in enumerate(images, start=1):
        print i, im.filename
        if (i > 1 and (i-1) % G == 0):
            x_offset = 0
            y_offset += 100

        new_im.paste(im, (x_offset, y_offset))
        x_offset += 100

    new_im.save('test.png')

def concat2(G):
    #images = [Image.open('raster/frame{:05d}.png'.format(x)) for x in range(1, (G * G)/2 + 1)] # [1, 801)
    images = [Image.open('raster/frame{:05d}.png'.format(x)) for x in range((G * G) / 2 + 1, G * G + 1)] #[801, 1601)
    print len(images)

    total_width = G * 100
    total_height = G * 50

    new_im = Image.new('RGB', (total_width, total_height))

    x_offset = 0
    y_offset = 0
    for i, im in enumerate(images, start=1):
        print i, im.filename
        if (i > 1 and (i-1) % G == 0):
            x_offset = 0
            y_offset += 100

        new_im.paste(im, (x_offset, y_offset))
        x_offset += 100

    new_im.save('test2.png')


def concat_halves(G):
    images = [Image.open('test.png'), Image.open('test2.png')]

    total_width = G * 100
    total_height = G * 100

    new_im = Image.new('RGB', (total_width, total_height))

    new_im.paste(images[0], (0, 0))
    new_im.paste(images[1], (0, total_height / 2))

    new_im.save('raster.png')

def concat_all(G, sharpness):
    for i in range(1, G + 1):
        concat_row(G, sharpness, i)
    concat_rows(G, sharpness)

def concat_row(G, sharpness, row):
    images = [Image.open('raster/frame{:05d}.png'.format(x)) for x in range((row - 1) * G + 1, row * G + 1)]
    print len(images)

    total_width = G * sharpness
    total_height = 1 * sharpness

    new_im = Image.new('RGB', (total_width, total_height))

    x_offset = 0
    for i, im in enumerate(images, start=1):
        print i, im.filename

        new_im.paste(im, (x_offset, 0))
        x_offset += sharpness

    new_im.save('raster/row{:03d}.png'.format(row))

def concat_rows(G, sharpness):
    images = [Image.open('raster/row{:03d}.png'.format(row)) for row in range(1, G + 1)]

    total_width = G * sharpness
    total_height = G * sharpness

    new_im = Image.new('RGB', (total_width, total_height))

    y_offset = 0
    for i, im in enumerate(images, start=1):
        print i, im.filename

        new_im.paste(im, (0, y_offset))
        y_offset += sharpness

    new_im.save('raster{}.png'.format(G))
