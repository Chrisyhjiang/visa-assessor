# Stage 1: Build the React app
FROM node:20-alpine as build
WORKDIR /app
COPY package.json package-lock.json* tsconfig.json ./
COPY public ./public
COPY src ./src
RUN npm install && npm run build

# Stage 2: Serve the build with a static server
FROM nginx:alpine
COPY --from=build /app/build /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"] 