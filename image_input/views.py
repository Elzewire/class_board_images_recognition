import datetime

import cv2
from PIL import Image
from django.http import JsonResponse
from django.shortcuts import render
from django.views.generic import TemplateView

from image_input.models import Entry, WEEKDAY_MAP
from recognition.decoder import dict_decode
from recognition.main import predict, predict_ru
from recognition.preprocess import prop_resize_pil
from recognition.segmentation import segmentation
from vec_search.metadata import get_meta
from vec_search.search import vector_search


class IndexView(TemplateView):
    template_name = "img_input.html"


def resize_view(request):
    if request.method == "GET":
        return render(request, "img_input.html")
    else:
        # get file from request.
        img = Image.open(request.FILES['image'])
        w, h = img.size
        path = 'static/resized.png'
        if w > 1280 or h > 720:
            res = prop_resize_pil(img, (1280, 720))
            res.save(path)
        else:
            img.save(path)

        return JsonResponse({'image': path})


def predict_view(request):
    data = {}
    if request.method == "GET":
        x = int(float(request.GET['x']))
        y = int(float(request.GET['y']))
        w = int(float(request.GET['w']))
        h = int(float(request.GET['h']))

        disciplines = None

        # Поиск по метаданным
        meta = get_meta('static/resized.png')
        if meta:
            dt = datetime.datetime.strptime(meta['DateTimeOriginal'], '%Y:%m:%d %H:%M:%S')
            if dt.month < 6:
                year = '{}, 2s'.format(dt.year - 1)
            else:
                year = '{}, 1s'.format(dt.year)

            entries = Entry.objects.filter(timetable__year=year, day=WEEKDAY_MAP[dt.weekday()])

            result = None
            if entries:
                # Ищем точные попадания
                for e in entries:
                    start_time = datetime.datetime.strptime(e.time.split('-')[0], '%H:%M')
                    end_time = datetime.datetime.strptime(e.time.split('-')[1], '%H:%M')

                    if start_time.time() <= dt.time() <= end_time.time():
                        result = e

                # Если ничего, то ищем ближайщую закончившуюся пару
                if not result:
                    min_time = 10
                    for e in entries:
                        end_time = datetime.datetime.strptime(e.time.split('-')[1], '%H:%M')

                        if abs(end_time.time().hour - dt.time().hour) < min_time:
                            min_time = abs(end_time.time().hour - dt.time().hour)
                            result = e

            if result:
                data['dt'] = dt.__str__()
                data['meta'] = result.disciplines.all()
                disciplines = result.disciplines.all()

        # Распознавание содержимого
        img = cv2.imread('static/resized.png')
        segments = segmentation(img[y:y+h, x:x+w])
        #new = img.copy()
        images = []
        #for k, s in enumerate(segments):
            #cv2.imshow('seg', s[0])
            #cv2.waitKey()
            #cv2.destroyAllWindows()

        labels_en = dict_decode(predict([s[0] for s in segments]), lang='en')
        labels_ru = dict_decode(predict_ru([s[0] for s in segments]), lang='ru')

        data['pred_en'] = labels_en
        data['pred_ru'] = labels_ru

        # Векторный поиск
        q = ' '.join(labels_ru)
        results = vector_search(q)
        print(results)
        data['results'] = [(a, b, 'static/РПД/{}'.format(c)) for a, b, c in results]

        results = [b.strip('\n') for a, b, c in results]
        if results and disciplines:
            probable = []
            for d in disciplines:
                if d.name in results:
                    probable.append(d.name)

            print(probable)

            data['probable'] = probable

    return render(request, "results.html", data)
