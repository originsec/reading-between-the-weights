FROM alpine:3.22
RUN apk add --no-cache bash \
    && mkdir -p /opt/fixture \
    && apk fetch --no-cache --output /opt/fixture alpine-baselayout-data \
    && mv /opt/fixture/alpine-baselayout-data-*.apk /opt/fixture/package.apk
