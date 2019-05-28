# get the k largest numbers in an array
# runtime: O(k*log(n))


def heapify(arr, size, root):
    largest = root
    left = 2*root
    right = 2*root + 1

    if left <= size and arr[left] > arr[largest]:
        largest = left

    if right <= size and arr[right] > arr[largest]:
        largest = right

    if largest != root:
        arr[root], arr[largest] = arr[largest], arr[root]

        heapify(arr, size, largest)


def get_k_largest(arr, k):
    n = len(arr)
    # build heap in O(n) runtime
    # proof: https://www.geeksforgeeks.org/time-complexity-of-building-a-heap/
    for i in range((n+1)//2-1, 0, -1):
        heapify(arr, n, i)

    for _ in range(k):
        print(arr[1])
        arr.pop(1)
        heapify(arr, len(arr) - 1, 1)  # rebuild heap from root in O(log(n))


if __name__ == '__main__':
    a = [None, 2, 5, 8, 7, 3, 6, 4, 9, 1]
    get_k_largest(a, 4)
