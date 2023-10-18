#!/usr/bin/env python3
"""Task 4"""

if __name__ == '__main__':
    from requests import get
    API = 'https://api.spacexdata.com/'
    LAUNCHES = 'v4/launches/'
    ROCKETS = 'v4/rockets/'

    rockets_count = {}
    rocket_names = {}

    launches = get(API + LAUNCHES).json()
    for launch in launches:

        # get rocket name from dictionary, else call API
        try:
            name = rocket_names[launch['rocket']]
            rockets_count[name] += 1    # add to count

        except KeyError:
            # get rocket name from API
            name = get(API + ROCKETS + launch['rocket']).json()['name']

            # add name to dict
            rocket_names[launch['rocket']] = name
            rockets_count[name] = 1     # create dict entry and set to 1

    # sort alphabetically into list of tuples
    rockets_count = sorted(rockets_count.items(), key=lambda d: d[0])

    # sort list by value and print
    for rocket in sorted(rockets_count, key=lambda d: d[1], reverse=True):
        print('{}: {}'.format(rocket[0], rocket[1]))
