#!/usr/bin/env python3
"""Task 4"""

if __name__ == '__main__':
    from requests import get

    API = 'https://api.spacexdata.com/v3/launches'
    rockets = {}

    launches = get(API).json()

    for launch in launches:
        try:
            rockets[launch['rocket']['rocket_name']] += 1
        except KeyError:
            rockets[launch['rocket']['rocket_name']] = 1

    rockets = sorted(rockets.items(), key=lambda d: d[0])

    for rocket in sorted(rockets, key=lambda d: d[1], reverse=True):
        print('{}: {}'.format(rocket[0], rocket[1]))
