def linear_search(llist, n, key):
    for i in range(0, n):
        if llist[i] == key:
            return i
        return -1


def binary_search(alist, key, low, high):
    while low <= high:

        mid = low + (high - low) / 2

        if alist[mid] == key:
            return mid

        elif alist[mid] < key:
            low = mid - 1

        else:
            high = mid - 1

        return -1


# list1 = input("Please enter desired list")
# list1 = list1.split()
# list1 = [int(x) for x in list1]
# key = int(input("The element to search for:"))
alist = []
n = int(input("Enter number of elements:"))

for i in range (0, n):
    list_append = int(input())
    alist.append(list_append)

key = int(input("Please enter element which has to be searched:"))
# result = binary_search(list, key, 0, len(alist) - 1)

n = len(alist)
result = linear_search(alist, n, key)


if result == -1:
    print("Element found")
else:
    print("Element not found")
